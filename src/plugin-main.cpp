// include opencv headers first to avoid conflicts
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>

// rest
#include <obs-module.h>
#include <util/platform.h>
#include <util/threading.h>
#include <graphics/image-file.h>

using namespace cv;

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-dualcam-switcher", "en-US")

struct camstate {
	float eye_dist_from_center;
	bool has_eyes;
};

// Plugin data structure
struct dualcam_switcher {
	obs_source_t *context;           // Our source
	obs_source_t *camera1;           // Reference to camera 1
	obs_source_t *camera2;           // Reference to camera 2
	
	char *camera1_name;
	char *camera2_name;
	
	int active_camera;               // 1 or 2
	bool manual_mode;                // Manual vs auto switching
	
	// Face detection state (to be implemented)
	pthread_t detection_thread;
	bool thread_running;
	volatile bool stop_thread;

	struct camstate cam1state;
	struct camstate cam2state;

	float switch_timer; // threshold
	int pending_camera; // cam that wants to be active
	float hysteresis_thresh;
	
	uint32_t width;
	uint32_t height;

	// Haar Cascade
	CascadeClassifier *eye_cascade;
	pthread_mutex_t state_mutex;

    // ADD: Frame buffers for thread processing
	// Double buffering for thread safety
	Mat cam1_frame_current;
	Mat cam2_frame_current;
	Mat cam1_frame_next;
	Mat cam2_frame_next;
	bool frames_ready;
	pthread_mutex_t frame_mutex;
	int frame_capture_counter;

	gs_texrender_t *texrender;
    gs_stagesurf_t *stage;
    uint32_t last_w, last_h;
};

// --- HELPER FUNCTIONS ---

static inline const char *get_cascade_path(const char *filename)
{
    static char path[512];
    const char *data_path = obs_get_module_data_path(obs_current_module());
    snprintf(path, sizeof(path), "%s/%s", data_path, filename);
    return path;
}

static Mat obs_source_to_mat(struct dualcam_switcher *context, obs_source_t *source)
{
    if (!source) return Mat();

    uint32_t width = obs_source_get_width(source);
    uint32_t height = obs_source_get_height(source);
    if (width == 0 || height == 0) return Mat();

    // Downscale targets
    uint32_t sw = 320;
    uint32_t sh = (height * sw) / width;

    // Re-create resources only if size changed or they don't exist
    if (!context->texrender || context->last_w != sw || context->last_h != sh) {
        if (context->texrender) gs_texrender_destroy(context->texrender);
        if (context->stage) gs_stagesurface_destroy(context->stage);

        context->texrender = gs_texrender_create(GS_BGRA, GS_ZS_NONE);
        context->stage = gs_stagesurface_create(sw, sh, GS_BGRA);
        context->last_w = sw;
        context->last_h = sh;
    }

    Mat result;
    gs_texrender_reset(context->texrender);
    
    if (gs_texrender_begin(context->texrender, sw, sh)) {
        struct vec4 clear_color = {0};
        gs_clear(GS_CLEAR_COLOR, &clear_color, 0.0f, 0);
        gs_ortho(0.0f, (float)sw, 0.0f, (float)sh, -100.0f, 100.0f);
        
        obs_source_video_render(source);
        gs_texrender_end(context->texrender);
        
        gs_texture_t *tex = gs_texrender_get_texture(context->texrender);
        if (tex) {
            gs_stage_texture(context->stage, tex);
            
            uint8_t *data;
            uint32_t linesize;
            if (gs_stagesurface_map(context->stage, &data, &linesize)) {
                // Create a temporary wrapper, then CLONE it to result 
                // so the data persists after unmap
                Mat frame(sh, sw, CV_8UC4, data, linesize);
                cvtColor(frame, result, COLOR_BGRA2BGR);
                gs_stagesurface_unmap(context->stage);
            }
        }
    }
    return result; // Returns a BGR Mat with its own allocated memory
}


static void process_camera_mat(const Mat &frame, CascadeClassifier *cascade, struct camstate *state)
{
    if (frame.empty() || !cascade || cascade->empty()) {
        state->has_eyes = false;
        return;
    }

    Mat gray;
    cvtColor(frame, gray, COLOR_BGR2GRAY);
    equalizeHist(gray, gray);

    std::vector<Rect> eyes;
    cascade->detectMultiScale(gray, eyes, 1.1, 3, 0, Size(30, 30));

    blog(LOG_INFO, "[DualCam] Detected %zu eyes", eyes.size());

    if (eyes.size() > 0) {
        float eye_x = (eyes[0].x + eyes[0].width / 2.0f) / (float)frame.cols;
        float eye_y = (eyes[0].y + eyes[0].height / 2.0f) / (float)frame.rows;
        
        float dx = eye_x - 0.5f;
        float dy = eye_y - 0.5f;
        
        state->eye_dist_from_center = sqrtf(dx*dx + dy*dy);
        state->has_eyes = true;
        
        blog(LOG_INFO, "[DualCam] Eye found, distance from center: %f", state->eye_dist_from_center);
    } else {
        state->has_eyes = false;
    }
}


static void *face_detection_thread(void *data)
{
    struct dualcam_switcher *context = (struct dualcam_switcher *)data;
    blog(LOG_INFO, "[DualCam] Detection thread started");
    
	while (!context->stop_thread) {
		Mat local_1, local_2;
		bool has_work = false;

		pthread_mutex_lock(&context->frame_mutex);
		if (context->frames_ready) {
			local_1 = std::move(context->cam1_frame_next);
			local_2 = std::move(context->cam2_frame_next);
			context->frames_ready = false;
			has_work = true;
		}
		pthread_mutex_unlock(&context->frame_mutex);

		if (has_work) {
			pthread_mutex_lock(&context->state_mutex);
			process_camera_mat(local_1, context->eye_cascade, &context->cam1state);
			process_camera_mat(local_2, context->eye_cascade, &context->cam2state);
			pthread_mutex_unlock(&context->state_mutex);
		}
		os_sleep_ms(100);
	}
    
    blog(LOG_INFO, "[DualCam] Detection thread stopped");
    return NULL;
}

// Forward declarations
// https://docs.obsproject.com/reference-sources
static const char *dualcam_get_name(void *unused);
static void *dualcam_create(obs_data_t *settings, obs_source_t *source); // allocates
static void dualcam_destroy(void *data);
static void dualcam_update(void *data, obs_data_t *settings); // called when settings change
static void dualcam_video_render(void *data, gs_effect_t *effect); // draws output
static void dualcam_video_tick(void *data, float seconds); // called once per frame
static obs_properties_t *dualcam_properties(void *data); // creates UI for settings
static void dualcam_get_defaults(obs_data_t *settings);
static uint32_t dualcam_get_width(void *data);
static uint32_t dualcam_get_height(void *data);



/**
 * Convert OBS source frame to OpenCV Mat
 * 
 * Renders an OBS source into a texture, reads the pixel data,
 * and converts it to an OpenCV Mat in BGR format for processing.
 * 
 * @param source The OBS source to capture
 * @param out_width Pointer to store the frame width
 * @param out_height Pointer to store the frame height
 * @return OpenCV Mat containing the frame in BGR format, or empty Mat on failure
 */
static Mat obs_source_to_mat(obs_source_t *source);


/**
 * Process a single camera frame for eye detection
 * 
 * Takes a camera frame, converts it to grayscale, runs eye detection,
 * and updates the camera state with the results.
 * 
 * @param frame OpenCV Mat containing the camera frame in BGR format
 * @param width Frame width in pixels
 * @param height Frame height in pixels
 * @param cascade Pointer to the loaded Haar cascade classifier
 * @param state Pointer to the camera state structure to update
 * @param cam_number Camera number (for logging purposes)
 */
static void process_camera_mat(const Mat &frame, CascadeClassifier *cascade, struct camstate *state);

/**
 * Face detection thread function
 * 
 * Continuously captures frames from both cameras, runs eye detection,
 * and updates the camera states. Runs at approximately 30fps.
 * 
 * @param data Pointer to the dualcam_switcher context
 * @return NULL on thread exit
 */
static void *face_detection_thread(void *data);

// Plugin info structure
// sets all the functions listed above
struct obs_source_info dualcam_source_info = {
	.id = "dualcam_face_switcher",
	.type = OBS_SOURCE_TYPE_INPUT,
	.output_flags = OBS_SOURCE_VIDEO | OBS_SOURCE_CUSTOM_DRAW,
	
	.get_name = dualcam_get_name,
	.create = dualcam_create,
	.destroy = dualcam_destroy,
	.update = dualcam_update,
	.get_properties = dualcam_properties,
	.get_defaults = dualcam_get_defaults,
	.video_render = dualcam_video_render,
	.video_tick = dualcam_video_tick,
	.get_width = dualcam_get_width,
	.get_height = dualcam_get_height,
};

// Module load
bool obs_module_load(void)
{
	blog(LOG_INFO, "DualCam Face Switcher plugin loaded");
	// since this plugin will operate as a video source for something else
	// i.e. the output will be input into someplace (like the virtual camera)
	// we register source
	obs_register_source(&dualcam_source_info);
	return true;
}

void obs_module_unload(void)
{
	blog(LOG_INFO, "DualCam Face Switcher plugin unloaded");
}

// Get plugin name
static const char *dualcam_get_name(void *unused)
{
	UNUSED_PARAMETER(unused);
	return "Dual Camera Face Switcher";
}

// Helper: Add video sources to dropdown
static bool add_source_to_list(void *data, obs_source_t *source)
{
	// PROPERTY = UI ELEMENT
	// properties are used to enumerate available settings for an object
	// "typically this is used to generate user interface widgets but can do specific settings as well"
	// https://docs.obsproject.com/reference-properties
	//
	// here, prop is the actual dropdown as passed
	obs_property_t *prop = (obs_property_t *)data;
	// passing in another source and saving to caps
	uint32_t caps = obs_source_get_output_flags(source);
	
	// Only add video sources
	if ((caps & OBS_SOURCE_VIDEO) != 0) {
		const char *name = obs_source_get_name(source);
		obs_property_list_add_string(prop, name, name);
	}
	
	return true;
}

// Create properties UI
static obs_properties_t *dualcam_properties(void *data)
{
	UNUSED_PARAMETER(data);
	
	// https://docs.obsproject.com/reference-properties
	obs_properties_t *props = obs_properties_create();
	
	// Camera 1 selection
	// *cam1 IS A DROPDOWN MENU
	// https://docs.obsproject.com/reference-properties
	// - name
	// - description
	// - type (combo_type_list means not editable)
	// - format: combo_format_string means string list
	obs_property_t *cam1 = obs_properties_add_list(props, "camera1",
		"Camera 1", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	// obs_enum_sources enumerates sources and calls the callback function we pass into it
	// here, we have add_source_to_list
	// so when we enum the sources, we pass that source to add_source_to_list, along with cam1
	// and then it decides whether or not to add the source to the cam1 dropdown
	obs_enum_sources(add_source_to_list, cam1);
	
	// Camera 2 selection
	//
	obs_property_t *cam2 = obs_properties_add_list(props, "camera2",
		"Camera 2", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_STRING);
	obs_enum_sources(add_source_to_list, cam2);
	
	// Manual mode toggle
	obs_properties_add_bool(props, "manual_mode", "Manual Mode (disable auto-switching)");
	
	// Manual camera selection (when in manual mode)
	obs_property_t *manual_select = obs_properties_add_list(props, "manual_camera",
		"Active Camera", OBS_COMBO_TYPE_LIST, OBS_COMBO_FORMAT_INT);
	obs_property_list_add_int(manual_select, "Camera 1", 1);
	obs_property_list_add_int(manual_select, "Camera 2", 2);
	
	// Future: Add face detection sensitivity, switching delay, etc.
	
	return props;
}

// Set default values
static void dualcam_get_defaults(obs_data_t *settings)
{
	obs_data_set_default_bool(settings, "manual_mode", false);
	obs_data_set_default_int(settings, "manual_camera", 1);
	obs_data_set_default_double(settings, "hysteresis_thresh", 0.05);
}

// Create plugin instance
static void *dualcam_create(obs_data_t *settings, obs_source_t *source)
{
    struct dualcam_switcher *context = (struct dualcam_switcher *)bzalloc(sizeof(struct dualcam_switcher));
    context->context = source;
    context->active_camera = 1;

    pthread_mutex_init(&context->state_mutex, NULL);

    context->manual_mode = true;
    context->thread_running = false;
    context->stop_thread = false;
    context->width = 1920;
    context->height = 1080;

    context->eye_cascade = new CascadeClassifier();
    const char *path = get_cascade_path("haarcascades/haarcascade_eye.xml");
    blog(LOG_INFO, "=== CASCADE PATH: %s ===", path);
    if (!context->eye_cascade->load(path)) {
        blog(LOG_ERROR, "[DualCam] Failed to load cascade: %s", path);
    }

	pthread_mutex_init(&context->frame_mutex, NULL);
	context->frames_ready = false;

	context->frame_capture_counter = 0;

    
    blog(LOG_INFO, "DualCam switcher created");
    
    // ADD THIS LOG
    blog(LOG_INFO, "[DualCam] About to call dualcam_update from create...");
    
    // Apply initial settings
    dualcam_update(context, settings);
    
    return context;
}

// Destroy plugin instance
static void dualcam_destroy(void *data)
{
	struct dualcam_switcher *context = (struct dualcam_switcher *)data;
	
	// Stop detection thread if running
	if (context->thread_running) {
		context->stop_thread = true;
		pthread_join(context->detection_thread, NULL);
	}
	
	// Release camera sources
	if (context->camera1) {
		obs_source_release(context->camera1);
	}
	if (context->camera2) {
		obs_source_release(context->camera2);
	}
	
	// Free strings
	bfree(context->camera1_name);
	bfree(context->camera2_name);

	delete context->eye_cascade;
	pthread_mutex_destroy(&context->state_mutex);

	pthread_mutex_destroy(&context->frame_mutex);
	
	blog(LOG_INFO, "DualCam switcher destroyed");
	bfree(context);
}

// Update settings
static void dualcam_update(void *data, obs_data_t *settings)
{
    struct dualcam_switcher *context = (struct dualcam_switcher *)data;
    
    blog(LOG_INFO, "[DualCam] ========== UPDATE CALLED ==========");
    
    // Get camera names from settings
    const char *cam1_name = obs_data_get_string(settings, "camera1");
    const char *cam2_name = obs_data_get_string(settings, "camera2");
    
    context->manual_mode = obs_data_get_bool(settings, "manual_mode");
    blog(LOG_INFO, "[DualCam] Manual mode: %d", context->manual_mode);
    
    if (context->manual_mode) {
        blog(LOG_INFO, "[DualCam] In manual mode, stopping thread if running");
        if (context->thread_running) {
            context->stop_thread = true;
            pthread_join(context->detection_thread, NULL);
            context->thread_running = false;
        }
        context->active_camera = (int)obs_data_get_int(settings, "manual_camera");
    } else if (!context->thread_running) {
        blog(LOG_INFO, "[DualCam] AUTO MODE - Starting detection thread...");
        context->stop_thread = false;
        int result = pthread_create(&context->detection_thread, NULL, face_detection_thread, context);
        context->thread_running = (result == 0);
        blog(LOG_INFO, "[DualCam] pthread_create result: %d, thread_running: %d", result, context->thread_running);
    } else {
        blog(LOG_INFO, "[DualCam] Thread already running");
    }
    
    // Update camera 1 reference
    if (cam1_name && strlen(cam1_name) > 0) {
        if (context->camera1) {
            obs_source_release(context->camera1);
        }
        context->camera1 = obs_get_source_by_name(cam1_name);
        
        bfree(context->camera1_name);
        context->camera1_name = bstrdup(cam1_name);
        
        blog(LOG_INFO, "Camera 1 set to: %s (source=%p)", cam1_name, context->camera1);
    }
    
    // Update camera 2 reference
    if (cam2_name && strlen(cam2_name) > 0) {
        if (context->camera2) {
            obs_source_release(context->camera2);
        }
        context->camera2 = obs_get_source_by_name(cam2_name);
        
        bfree(context->camera2_name);
        context->camera2_name = bstrdup(cam2_name);
        
        blog(LOG_INFO, "Camera 2 set to: %s (source=%p)", cam2_name, context->camera2);
    }
    
    // Update dimensions from active camera
    obs_source_t *active = (context->active_camera == 1) ? context->camera1 : context->camera2;
    if (active) {
        context->width = obs_source_get_width(active);
        context->height = obs_source_get_height(active);
    }
    
    blog(LOG_INFO, "[DualCam] ========== UPDATE COMPLETE ==========");
}

// Called every frame before rendering
static void dualcam_video_tick(void *data, float seconds)
{
    struct dualcam_switcher *context = (struct dualcam_switcher *)data;
    
    if (context->manual_mode) return;
    
    // NO GRAPHICS CALLS HERE - just switching logic
    pthread_mutex_lock(&context->state_mutex);
    
    int winner = 1;
    if (context->cam1state.has_eyes && context->cam2state.has_eyes) {
        float diff = context->cam1state.eye_dist_from_center - context->cam2state.eye_dist_from_center;
        if (diff > context->hysteresis_thresh) winner = 2;
        else if (diff < -context->hysteresis_thresh) winner = 1;
        else winner = context->active_camera;
    } else if (context->cam2state.has_eyes) {
        winner = 2;
    }

    pthread_mutex_unlock(&context->state_mutex);

    if (winner != context->active_camera) {
        if (winner == context->pending_camera) {
            context->switch_timer += seconds;
            if (context->switch_timer >= 2.0f) {
                blog(LOG_INFO, "[DualCam] SWITCHING to camera %d", winner);
                context->active_camera = winner;
                context->switch_timer = 0.0f;
            }
        } else {
            context->pending_camera = winner;
            context->switch_timer = 0.0f;
        }
    } else {
        context->switch_timer = 0.0f;
    }
}

// Render the active camera
static void dualcam_video_render(void *data, gs_effect_t *effect)
{
	struct dualcam_switcher *context = (struct dualcam_switcher *)data;
	UNUSED_PARAMETER(effect);
	
	obs_source_t *active_source = (context->active_camera == 1) 
		? context->camera1 
		: context->camera2;
	
	if (!active_source) {
		blog(LOG_WARNING, "No active camera source");
		return;
	}
	
	if (active_source == context->context) {
		blog(LOG_ERROR, "Cannot render self - circular reference detected!");
		return;
	}

	if (!obs_source_active(active_source)) {
		blog(LOG_DEBUG, "DualCam: source is not active!");
		return;
	}
	
	// Render the active camera's output FIRST
	obs_source_video_render(active_source);
	
	// THEN capture frames for detection
	if (!context->manual_mode) {
		static int frame_count = 0;
		if (++frame_count % 10 == 0) {
			// Capture frames locally first to keep the lock duration short
			Mat m1 = obs_source_to_mat(context, context->camera1);
			Mat m2 = obs_source_to_mat(context, context->camera2);

			pthread_mutex_lock(&context->frame_mutex);
			if (!context->frames_ready) {
				context->cam1_frame_next = std::move(m1);
				context->cam2_frame_next = std::move(m2);
				context->frames_ready = true;
			}
			pthread_mutex_unlock(&context->frame_mutex);
		}
	}
}

// Get output width
static uint32_t dualcam_get_width(void *data)
{
	struct dualcam_switcher *context = (struct dualcam_switcher *)data;
	return context->width;
}

// Get output height
static uint32_t dualcam_get_height(void *data)
{
	struct dualcam_switcher *context = (struct dualcam_switcher *)data;
	return context->height;
}
