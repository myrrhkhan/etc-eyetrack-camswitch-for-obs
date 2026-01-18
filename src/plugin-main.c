#include <obs-module.h>
#include <util/platform.h>
#include <util/threading.h>

OBS_DECLARE_MODULE()
OBS_MODULE_USE_DEFAULT_LOCALE("obs-dualcam-switcher", "en-US")

struct camstate {
	float eye_dist_from_center;
	bool has_eyes;
}

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
};

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
	obs_data_set_default_bool(settings, "manual_mode", true);
	obs_data_set_default_int(settings, "manual_camera", 1);
}

// Create plugin instance
static void *dualcam_create(obs_data_t *settings, obs_source_t *source)
{
	// this struct is the same as above
	struct dualcam_switcher *context = bzalloc(sizeof(struct dualcam_switcher));
	context->context = source;
	context->active_camera = 1;
	context->manual_mode = true;
	context->thread_running = false;
	context->stop_thread = false;
	context->width = 1920;
	context->height = 1080;
	
	blog(LOG_INFO, "DualCam switcher created");
	
	// Apply initial settings
	dualcam_update(context, settings);
	
	return context;
}

// Destroy plugin instance
static void dualcam_destroy(void *data)
{
	struct dualcam_switcher *context = data;
	
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
	
	blog(LOG_INFO, "DualCam switcher destroyed");
	bfree(context);
}

// Update settings
static void dualcam_update(void *data, obs_data_t *settings)
{
	struct dualcam_switcher *context = data;
	
	// Get camera names from settings
	const char *cam1_name = obs_data_get_string(settings, "camera1");
	const char *cam2_name = obs_data_get_string(settings, "camera2");
	
	context->manual_mode = obs_data_get_bool(settings, "manual_mode");
	
	if (context->manual_mode) {
		context->active_camera = (int)obs_data_get_int(settings, "manual_camera");
	}
	
	// Update camera 1 reference
	if (cam1_name && strlen(cam1_name) > 0) {
		if (context->camera1) {
			obs_source_release(context->camera1);
		}
		context->camera1 = obs_get_source_by_name(cam1_name);
		
		bfree(context->camera1_name);
		context->camera1_name = bstrdup(cam1_name);
		
		blog(LOG_INFO, "Camera 1 set to: %s", cam1_name);
	}
	
	// Update camera 2 reference
	if (cam2_name && strlen(cam2_name) > 0) {
		if (context->camera2) {
			obs_source_release(context->camera2);
		}
		context->camera2 = obs_get_source_by_name(cam2_name);
		
		bfree(context->camera2_name);
		context->camera2_name = bstrdup(cam2_name);
		
		blog(LOG_INFO, "Camera 2 set to: %s", cam2_name);
	}
	
	// Update dimensions from active camera
	obs_source_t *active = (context->active_camera == 1) ? context->camera1 : context->camera2;
	if (active) {
		context->width = obs_source_get_width(active);
		context->height = obs_source_get_height(active);
	}
}

// Called every frame before rendering
static void dualcam_video_tick(void *data, float seconds)
{
	struct dualcam_switcher *context = data;
	UNUSED_PARAMETER(seconds);
	
	// In auto mode, this is where you'd check face detection results
	// and switch cameras based on that
	
	if (!context->manual_mode) {
		// TODO: Check face detection state and switch cameras
		// For now, just alternate every 5 seconds for testing
		static float time_elapsed = 0;
		time_elapsed += seconds;
		if (time_elapsed > 5.0f) {
		    context->active_camera = (context->active_camera == 1) ? 2 : 1;
		    time_elapsed = 0;
		}
	}
}

// Render the active camera
static void dualcam_video_render(void *data, gs_effect_t *effect)
{
	struct dualcam_switcher *context = data;
	UNUSED_PARAMETER(effect);
	
	obs_source_t *active_source = (context->active_camera == 1) 
		? context->camera1 
		: context->camera2;
	
	if (!active_source) {
		blog(LOG_WARNING, "No active camera source");
		return;
	}
	
	// prevent circular reference
	if (active_source == context->context) {
		blog(LOG_ERROR, "Cannot render self - circular reference detected!");
		return;
	}

	if (!obs_source_active(active_source)) {
		blog(LOG_DEBUG, "DualCam: source is not active!");
		return;
	}

	// Log what we're rendering (remove this after debugging)
	const char *source_name = obs_source_get_name(active_source);
	blog(LOG_DEBUG, "DualCam: Rendering camera %d (%s)", 
	     context->active_camera, source_name ? source_name : "unknown");
	
	// Render the active camera's output
	obs_source_video_render(active_source);
}

// Get output width
static uint32_t dualcam_get_width(void *data)
{
	struct dualcam_switcher *context = data;
	return context->width;
}

// Get output height
static uint32_t dualcam_get_height(void *data)
{
	struct dualcam_switcher *context = data;
	return context->height;
}

/* 
 * TODO: Face detection implementation
 * 
 * 1. Create a detection thread function:
 *    void *face_detection_thread(void *data)
 * 
 * 2. In the thread:
 *    - Get frames from both cameras
 *    - Run face detection (OpenCV, MediaPipe, etc.)
 *    - Update a shared state variable with results
 *    - Sleep briefly to avoid excessive CPU usage
 * 
 * 3. In dualcam_video_tick:
 *    - Read the detection results
 *    - Switch active_camera based on which has faces
 *    - Add hysteresis to prevent rapid switching
 */
