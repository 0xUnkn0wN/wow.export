/*!
	wow.export (https://github.com/Kruithne/wow.export)
	Authors: Kruithne <kruithne@gmail.com>
	License: MIT
*/

const fs = require('fs');
const path = require('path');
const constants = require('../constants');
const log = require('../log');
const ShaderProgram = require('./gl/ShaderProgram');

const SHADER_MANIFEST = {
	m2: { vert: 'm2.vertex.shader', frag: 'm2.fragment.shader', wgsl: 'm2.wgsl' },
	wmo: { vert: 'wmo.vertex.shader', frag: 'wmo.fragment.shader', wgsl: 'wmo.wgsl' },
	adt: { vert: 'adt.vertex.shader', frag: 'adt.fragment.shader', wgsl: 'adt.wgsl' },
	adt_old: { vert: 'adt.vertex.shader', frag: 'adt.fragment.old.shader', wgsl: 'adt_old.wgsl' },
	char: { vert: 'char.vertex.shader', frag: 'char.fragment.shader', wgsl: 'char.wgsl' }
};

// cached shader source text
const source_cache = new Map();

/**
 * Load shader source from disk (cached)
 * @param {string} name
 * @returns {{ vert: string, frag: string }}
 */
function get_source(name) {
	if (source_cache.has(name))
		return source_cache.get(name);

	const manifest = SHADER_MANIFEST[name];
	if (!manifest)
		throw new Error(`Unknown shader: ${name}`);

	const shader_path = constants.SHADER_PATH;
	const vert = fs.readFileSync(path.join(shader_path, manifest.vert), 'utf8');
	const frag = fs.readFileSync(path.join(shader_path, manifest.frag), 'utf8');

	const sources = { vert, frag };
	source_cache.set(name, sources);
	return sources;
}

/**
 * Load WGSL shader source from disk (cached).
 * Returns null if the WGSL file does not exist.
 * @param {string} name
 * @returns {string|null}
 */
function get_wgsl_source(name) {
	const cache_key = name + '_wgsl';
	if (source_cache.has(cache_key))
		return source_cache.get(cache_key);

	const manifest = SHADER_MANIFEST[name];
	if (!manifest || !manifest.wgsl)
		return null;

	const shader_path = constants.SHADER_PATH;
	const wgsl_path = path.join(shader_path, manifest.wgsl);

	try {
		const wgsl = fs.readFileSync(wgsl_path, 'utf8');
		source_cache.set(cache_key, wgsl);
		return wgsl;
	} catch {
		// WGSL file doesn't exist yet
		return null;
	}
}

// active shader programs grouped by shader name
// Map<name, Set<ShaderProgram>>
const active_programs = new Map();

/**
 * Create and register a shader program
 * @param {GLContext|GPUContext} ctx
 * @param {string} name
 * @returns {ShaderProgram}
 */
function create_program(ctx, name) {
	let program;

	if (ctx.is_webgpu) {
		const wgsl = get_wgsl_source(name);
		if (!wgsl)
			throw new Error(`No WGSL shader found for: ${name}. Create ${SHADER_MANIFEST[name]?.wgsl} in the shaders directory.`);

		program = new ShaderProgram(ctx, wgsl);
	} else {
		const sources = get_source(name);
		program = new ShaderProgram(ctx, sources.vert, sources.frag);
	}

	if (!program.is_valid())
		throw new Error(`Failed to compile shader: ${name}`);

	// track for hot-reload
	program._shader_name = name;

	if (!active_programs.has(name))
		active_programs.set(name, new Set());

	active_programs.get(name).add(program);

	return program;
}

/**
 * Unregister a shader program (call on dispose)
 * @param {ShaderProgram} program
 */
function unregister(program) {
	const name = program._shader_name;
	if (!name)
		return;

	const programs = active_programs.get(name);
	if (programs)
		programs.delete(program);
}

/**
 * Reload all shaders from disk
 */
function reload_all() {
	log.write('Reloading all shaders...');

	// clear source cache to force re-read from disk
	source_cache.clear();

	let success_count = 0;
	let fail_count = 0;

	for (const [name, programs] of active_programs) {
		if (programs.size === 0)
			continue;

		try {
			for (const program of programs) {
				let ok;
				if (program.ctx.is_webgpu) {
					const wgsl = get_wgsl_source(name);
					ok = wgsl ? program.recompile(wgsl) : false;
				} else {
					const sources = get_source(name);
					ok = program.recompile(sources.vert, sources.frag);
				}

				if (ok)
					success_count++;
				else {
					fail_count++;
					log.write(`Failed to recompile shader program: ${name}`);
				}
			}
		} catch (e) {
			fail_count += programs.size;
			log.write(`Failed to reload shader source: ${name} - ${e.message}`);
		}
	}

	log.write(`Shader reload complete: ${success_count} succeeded, ${fail_count} failed`);
}

/**
 * Get count of active programs for a shader
 * @param {string} name
 * @returns {number}
 */
function get_program_count(name) {
	const programs = active_programs.get(name);
	return programs ? programs.size : 0;
}

/**
 * Get total count of all active programs
 * @returns {number}
 */
function get_total_program_count() {
	let count = 0;
	for (const programs of active_programs.values())
		count += programs.size;

	return count;
}

module.exports = {
	SHADER_MANIFEST,
	get_source,
	get_wgsl_source,
	create_program,
	unregister,
	reload_all,
	get_program_count,
	get_total_program_count
};
