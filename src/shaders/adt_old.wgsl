// ADT terrain shader with simple alpha blending (WebGPU / WGSL)
// Translated from adt.vertex.shader + adt.fragment.old.shader

struct Uniforms {
	translation: vec2<f32>,
	resolution: vec2<f32>,
	zoom: f32,
	layer_count: i32,
	_pad0: f32,
	_pad1: f32,
	// packed as vec4 pairs: [0..3] and [4..7]
	layer_scales_0: vec4<f32>,
	layer_scales_1: vec4<f32>,
	height_scales_0: vec4<f32>,
	height_scales_1: vec4<f32>,
	height_offsets_0: vec4<f32>,
	height_offsets_1: vec4<f32>,
	diffuse_indices_0: vec4<f32>,
	diffuse_indices_1: vec4<f32>,
	height_indices_0: vec4<f32>,
	height_indices_1: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;

@group(1) @binding(0) var t_diffuse_layers: texture_2d_array<f32>;
@group(1) @binding(1) var s_diffuse_layers: sampler;
@group(1) @binding(2) var t_height_layers: texture_2d_array<f32>;
@group(1) @binding(3) var s_height_layers: sampler;
@group(1) @binding(4) var t_alpha0: texture_2d<f32>;
@group(1) @binding(5) var s_alpha: sampler;
@group(1) @binding(6) var t_alpha1: texture_2d<f32>;
@group(1) @binding(7) var t_alpha2: texture_2d<f32>;
@group(1) @binding(8) var t_alpha3: texture_2d<f32>;
@group(1) @binding(9) var t_alpha4: texture_2d<f32>;
@group(1) @binding(10) var t_alpha5: texture_2d<f32>;
@group(1) @binding(11) var t_alpha6: texture_2d<f32>;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) texcoord: vec2<f32>,
	@location(2) color: vec4<f32>,
};

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) texcoord: vec2<f32>,
	@location(1) color: vec4<f32>,
};

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
	var out: VertexOutput;
	let pos = vec2<f32>(in.position.x, in.position.z) + u.translation;
	let zero_to_one = pos / u.resolution;
	let zero_to_two = zero_to_one * 2.0;
	let clip_space = zero_to_two - 1.0;
	out.position = vec4<f32>(clip_space * vec2<f32>(1.0, -1.0), 0.0, u.zoom);
	out.texcoord = in.texcoord * vec2<f32>(16.0, -16.0);
	out.color = in.color;
	return out;
}

// helper to index packed vec4 pairs
fn get_layer_scale(i: i32) -> f32 {
	if (i < 4) { return u.layer_scales_0[i]; }
	return u.layer_scales_1[i - 4];
}

fn get_diffuse_index(i: i32) -> i32 {
	if (i < 4) { return i32(u.diffuse_indices_0[i]); }
	return i32(u.diffuse_indices_1[i - 4]);
}

fn sample_alpha(idx: i32, uv: vec2<f32>) -> f32 {
	switch idx {
		case 0 { return textureSample(t_alpha0, s_alpha, uv).r; }
		case 1 { return textureSample(t_alpha1, s_alpha, uv).r; }
		case 2 { return textureSample(t_alpha2, s_alpha, uv).r; }
		case 3 { return textureSample(t_alpha3, s_alpha, uv).r; }
		case 4 { return textureSample(t_alpha4, s_alpha, uv).r; }
		case 5 { return textureSample(t_alpha5, s_alpha, uv).r; }
		case 6 { return textureSample(t_alpha6, s_alpha, uv).r; }
		default { return 0.0; }
	}
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	let alpha_uv = in.texcoord % vec2<f32>(1.0);

	// read alpha values for each layer
	var alphas: array<f32, 8>;
	alphas[0] = 1.0;
	for (var i = 1; i < 8; i++) {
		if (i < u.layer_count) {
			alphas[i] = sample_alpha(i - 1, alpha_uv);
		} else {
			alphas[i] = 0.0;
		}
	}

	// simple alpha blending without height-based weighting
	// base layer
	let tc0 = in.texcoord * (8.0 / get_layer_scale(0));
	var final_color = textureSample(t_diffuse_layers, s_diffuse_layers, tc0, get_diffuse_index(0)).rgb;

	// blend layers 1-7 on top
	for (var i = 1; i < 8; i++) {
		if (i >= u.layer_count) {
			break;
		}
		let tc = in.texcoord * (8.0 / get_layer_scale(i));
		let layer_color = textureSample(t_diffuse_layers, s_diffuse_layers, tc, get_diffuse_index(i)).rgb;
		final_color = mix(final_color, layer_color, alphas[i]);
	}

	return vec4<f32>(final_color * in.color.rgb * 2.0, 1.0);
}
