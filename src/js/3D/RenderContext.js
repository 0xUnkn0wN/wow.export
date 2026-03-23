/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const GLContext = require('./gl/GLContext');
const GPUContext = require('./gpu/GPUContext');
const core = require('../core');

const BACKEND_WEBGL2 = 'webgl2';
const BACKEND_WEBGPU = 'webgpu';

/**
 * Returns the configured render backend, falling back to webgl2.
 * @returns {string}
 */
function get_backend() {
	return core.view?.config?.renderBackend ?? BACKEND_WEBGL2;
}

/**
 * Check if WebGPU is available in this environment.
 * @returns {boolean}
 */
function is_webgpu_available() {
	return typeof navigator !== 'undefined' && !!navigator.gpu;
}

/**
 * Create a render context for the given canvas.
 * Returns a GLContext or GPUContext depending on configuration and availability.
 *
 * @param {HTMLCanvasElement} canvas
 * @param {object} [options]
 * @returns {Promise<GLContext|GPUContext>}
 */
async function create_render_context(canvas, options = {}) {
	const backend = get_backend();

	if (backend === BACKEND_WEBGPU) {
		if (!is_webgpu_available()) {
			console.warn('WebGPU requested but not available, falling back to WebGL2');
			return new GLContext(canvas, options);
		}

		const ctx = new GPUContext(canvas, options);
		await ctx.init();
		return ctx;
	}

	return new GLContext(canvas, options);
}

module.exports = {
	create_render_context,
	is_webgpu_available,
	get_backend,
	BACKEND_WEBGL2,
	BACKEND_WEBGPU
};
