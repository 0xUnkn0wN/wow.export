// M2 model shader (WebGPU / WGSL)
// Translated from m2.vertex.shader + m2.fragment.shader

struct Uniforms {
	view_matrix: mat4x4<f32>,
	projection_matrix: mat4x4<f32>,
	model_matrix: mat4x4<f32>,
	view_up: vec4<f32>,
	time: f32,
	bone_count: i32,
	has_tex_matrix1: i32,
	has_tex_matrix2: i32,
	tex_matrix1: mat4x4<f32>,
	tex_matrix2: mat4x4<f32>,
	vertex_shader_id: i32,
	pixel_shader_id: i32,
	blend_mode: i32,
	apply_lighting: i32,
	mesh_color: vec4<f32>,
	tex_sample_alpha: vec4<f32>,
	alpha_test: f32,
	wireframe: i32,
	_pad0: f32,
	_pad1: f32,
	wireframe_color: vec4<f32>,
	ambient_color: vec4<f32>,
	diffuse_color: vec4<f32>,
	light_dir: vec4<f32>,
	bone_matrices: array<mat4x4<f32>, 256>,
};

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(1) @binding(0) var t_texture1: texture_2d<f32>;
@group(1) @binding(1) var s_texture1: sampler;
@group(1) @binding(2) var t_texture2: texture_2d<f32>;
@group(1) @binding(3) var s_texture2: sampler;
@group(1) @binding(4) var t_texture3: texture_2d<f32>;
@group(1) @binding(5) var s_texture3: sampler;
@group(1) @binding(6) var t_texture4: texture_2d<f32>;
@group(1) @binding(7) var s_texture4: sampler;

struct VertexInput {
	@location(0) position: vec3<f32>,
	@location(1) normal: vec3<f32>,
	@location(2) bone_indices: vec4<u32>,
	@location(3) bone_weights: vec4<f32>,
	@location(4) texcoord: vec2<f32>,
};

struct VertexOutput {
	@builtin(position) position: vec4<f32>,
	@location(0) texcoord: vec2<f32>,
	@location(1) texcoord2: vec2<f32>,
	@location(2) texcoord3: vec2<f32>,
	@location(3) normal: vec3<f32>,
	@location(4) position_view: vec3<f32>,
	@location(5) edge_fade: f32,
};

fn calc_env_coord(pos_view: vec3<f32>, normal_view: vec3<f32>) -> vec2<f32> {
	let r = reflect(normalize(pos_view), normalize(normal_view));
	let m = 2.0 * sqrt(r.x * r.x + r.y * r.y + (r.z + 1.0) * (r.z + 1.0));
	return vec2<f32>(r.x / m + 0.5, r.y / m + 0.5);
}

fn calc_edge_fade(pos_view: vec3<f32>, normal_view: vec3<f32>) -> f32 {
	let view_dir = normalize(-pos_view);
	let n_dot_v = abs(dot(normalize(normal_view), view_dir));
	return clamp(n_dot_v * n_dot_v, 0.0, 1.0);
}

// compute transpose(inverse(mat3(m))) for normal transformation
fn normal_matrix(m: mat4x4<f32>) -> mat3x3<f32> {
	let c0 = m[0].xyz;
	let c1 = m[1].xyz;
	let c2 = m[2].xyz;
	let det = dot(c0, cross(c1, c2));
	let inv_det = 1.0 / det;
	return mat3x3<f32>(
		cross(c1, c2) * inv_det,
		cross(c2, c0) * inv_det,
		cross(c0, c1) * inv_det
	);
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
	var out: VertexOutput;

	// bone skinning
	var bone_transform = mat4x4<f32>(
		vec4<f32>(1,0,0,0), vec4<f32>(0,1,0,0),
		vec4<f32>(0,0,1,0), vec4<f32>(0,0,0,1)
	);

	if (u.bone_count > 0) {
		let total_weight = in.bone_weights.x + in.bone_weights.y + in.bone_weights.z + in.bone_weights.w;
		if (total_weight > 0.0) {
			bone_transform = in.bone_weights.x * u.bone_matrices[in.bone_indices.x]
				+ in.bone_weights.y * u.bone_matrices[in.bone_indices.y]
				+ in.bone_weights.z * u.bone_matrices[in.bone_indices.z]
				+ in.bone_weights.w * u.bone_matrices[in.bone_indices.w];
		}
	}

	let skinned_pos = bone_transform * vec4<f32>(in.position, 1.0);
	let world_pos = u.model_matrix * skinned_pos;
	let view_pos = u.view_matrix * world_pos;
	out.position = u.projection_matrix * view_pos;

	let nm = normal_matrix(u.view_matrix * u.model_matrix * bone_transform);
	let normal_view = normalize(nm * in.normal);

	out.normal = normal_view;
	out.position_view = view_pos.xyz;

	let env_coord = calc_env_coord(view_pos.xyz, normal_view);
	let edge_scan = calc_edge_fade(view_pos.xyz, normal_view);
	out.edge_fade = 1.0;

	let ident4 = mat4x4<f32>(
		vec4<f32>(1,0,0,0), vec4<f32>(0,1,0,0),
		vec4<f32>(0,0,1,0), vec4<f32>(0,0,0,1)
	);
	var tm1 = ident4;
	if (u.has_tex_matrix1 != 0) { tm1 = u.tex_matrix1; }
	var tm2 = ident4;
	if (u.has_tex_matrix2 != 0) { tm2 = u.tex_matrix2; }

	out.texcoord = in.texcoord;
	out.texcoord2 = vec2<f32>(0.0);
	out.texcoord3 = vec2<f32>(0.0);

	switch u.vertex_shader_id {
		case 0 { // Diffuse_T1
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
		}
		case 1 { // Diffuse_Env
			out.texcoord = env_coord;
		}
		case 2 { // Diffuse_T1_T2
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord2 = (tm2 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
		}
		case 3 { // Diffuse_T1_Env
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord2 = env_coord;
		}
		case 4 { // Diffuse_Env_T1
			out.texcoord = env_coord;
			out.texcoord2 = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
		}
		case 5 { // Diffuse_Env_Env
			out.texcoord = env_coord;
			out.texcoord2 = env_coord;
		}
		case 6 { // Diffuse_T1_Env_T1
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord2 = env_coord;
			out.texcoord3 = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
		}
		case 7 { // Diffuse_T1_T1
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord2 = out.texcoord;
		}
		case 8 { // Diffuse_T1_T1_T1
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord2 = out.texcoord;
			out.texcoord3 = out.texcoord;
		}
		case 9 { // Diffuse_EdgeFade_T1
			out.edge_fade = edge_scan;
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
		}
		case 10 { // Diffuse_T2
			out.texcoord = (tm2 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
		}
		case 11 { // Diffuse_T1_Env_T2
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord2 = env_coord;
			out.texcoord3 = (tm2 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
		}
		case 12 { // Diffuse_EdgeFade_T1_T2
			out.edge_fade = edge_scan;
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord2 = (tm2 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
		}
		case 13 { // Diffuse_EdgeFade_Env
			out.edge_fade = edge_scan;
			out.texcoord = env_coord;
		}
		case 14 { // Diffuse_T1_T2_T1
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord2 = (tm2 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord3 = out.texcoord;
		}
		case 15 { // Diffuse_T1_T2_T3
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord2 = (tm2 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord3 = in.texcoord;
		}
		case 16 { // Color_T1_T2_T3
			out.texcoord = (tm2 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
			out.texcoord2 = vec2<f32>(0.0);
			out.texcoord3 = in.texcoord;
		}
		case 17, 18 { // BW_Diffuse_T1, BW_Diffuse_T1_T2
			out.texcoord = (tm1 * vec4<f32>(in.texcoord, 0.0, 1.0)).xy;
		}
		default {
			out.texcoord = in.texcoord;
		}
	}

	return out;
}

// lighting
fn calc_lighting(color: vec3<f32>, normal: vec3<f32>) -> vec3<f32> {
	if (u.apply_lighting == 0) {
		return color;
	}
	let n = normalize(normal);
	let n_dot_l = max(dot(n, normalize(-u.light_dir.xyz)), 0.0);
	let ambient = u.ambient_color.xyz * color;
	let diffuse = u.diffuse_color.xyz * color * n_dot_l;
	return ambient + diffuse;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
	if (u.wireframe != 0) {
		return u.wireframe_color;
	}

	var uv1 = in.texcoord;
	var uv2 = in.texcoord2;
	var uv3 = in.texcoord3;

	if (u.pixel_shader_id == 26 || u.pixel_shader_id == 27 || u.pixel_shader_id == 28) {
		uv2 = uv1;
		uv3 = uv1;
	}

	let tex1 = textureSample(t_texture1, s_texture1, uv1);
	let tex2 = textureSample(t_texture2, s_texture2, uv2);
	let tex3 = textureSample(t_texture3, s_texture3, uv3);
	let tex4 = textureSample(t_texture4, s_texture4, in.texcoord2);

	let mesh_color = u.mesh_color.rgb;
	let mesh_opacity = u.mesh_color.a * in.edge_fade;

	var mat_diffuse = vec3<f32>(0.0);
	var specular = vec3<f32>(0.0);
	var discard_alpha = 1.0;
	var can_discard = false;

	switch u.pixel_shader_id {
		case 0 { // Combiners_Opaque
			mat_diffuse = mesh_color * tex1.rgb;
		}
		case 1 { // Combiners_Mod
			mat_diffuse = mesh_color * tex1.rgb;
			discard_alpha = tex1.a;
			can_discard = true;
		}
		case 2 { // Combiners_Opaque_Mod
			mat_diffuse = mesh_color * tex1.rgb * tex2.rgb;
			discard_alpha = tex2.a;
			can_discard = true;
		}
		case 3 { // Combiners_Opaque_Mod2x
			mat_diffuse = mesh_color * tex1.rgb * tex2.rgb * 2.0;
			discard_alpha = tex2.a * 2.0;
			can_discard = true;
		}
		case 4 { // Combiners_Opaque_Mod2xNA
			mat_diffuse = mesh_color * tex1.rgb * tex2.rgb * 2.0;
		}
		case 5 { // Combiners_Opaque_Opaque
			mat_diffuse = mesh_color * tex1.rgb * tex2.rgb;
		}
		case 6 { // Combiners_Mod_Mod
			mat_diffuse = mesh_color * tex1.rgb * tex2.rgb;
			discard_alpha = tex1.a * tex2.a;
			can_discard = true;
		}
		case 7 { // Combiners_Mod_Mod2x
			mat_diffuse = mesh_color * tex1.rgb * tex2.rgb * 2.0;
			discard_alpha = tex1.a * tex2.a * 2.0;
			can_discard = true;
		}
		case 8 { // Combiners_Mod_Add
			mat_diffuse = mesh_color * tex1.rgb;
			discard_alpha = tex1.a + tex2.a;
			can_discard = true;
			specular = tex2.rgb;
		}
		case 9 { // Combiners_Mod_Mod2xNA
			mat_diffuse = mesh_color * tex1.rgb * tex2.rgb * 2.0;
			discard_alpha = tex1.a;
			can_discard = true;
		}
		case 10 { // Combiners_Mod_AddNA
			mat_diffuse = mesh_color * tex1.rgb;
			discard_alpha = tex1.a;
			can_discard = true;
			specular = tex2.rgb;
		}
		case 11 { // Combiners_Mod_Opaque
			mat_diffuse = mesh_color * tex1.rgb * tex2.rgb;
			discard_alpha = tex1.a;
			can_discard = true;
		}
		case 12 { // Combiners_Opaque_Mod2xNA_Alpha
			mat_diffuse = mesh_color * mix(tex1.rgb * tex2.rgb * 2.0, tex1.rgb, vec3<f32>(tex1.a));
		}
		case 13 { // Combiners_Opaque_AddAlpha
			mat_diffuse = mesh_color * tex1.rgb;
			specular = tex2.rgb * tex2.a;
		}
		case 14 { // Combiners_Opaque_AddAlpha_Alpha
			mat_diffuse = mesh_color * tex1.rgb;
			specular = tex2.rgb * tex2.a * (1.0 - tex1.a);
		}
		case 15 { // Combiners_Opaque_Mod2xNA_Alpha_Add
			mat_diffuse = mesh_color * mix(tex1.rgb * tex2.rgb * 2.0, tex1.rgb, vec3<f32>(tex1.a));
			specular = tex3.rgb * tex3.a * u.tex_sample_alpha.z;
		}
		case 16 { // Combiners_Mod_AddAlpha
			mat_diffuse = mesh_color * tex1.rgb;
			discard_alpha = tex1.a;
			can_discard = true;
			specular = tex2.rgb * tex2.a;
		}
		case 17 { // Combiners_Mod_AddAlpha_Alpha
			mat_diffuse = mesh_color * tex1.rgb;
			discard_alpha = tex1.a + tex2.a * (0.3 * tex2.r + 0.59 * tex2.g + 0.11 * tex2.b);
			can_discard = true;
			specular = tex2.rgb * tex2.a * (1.0 - tex1.a);
		}
		case 18 { // Combiners_Opaque_Alpha_Alpha
			mat_diffuse = mesh_color * mix(mix(tex1.rgb, tex2.rgb, vec3<f32>(tex2.a)), tex1.rgb, vec3<f32>(tex1.a));
		}
		case 19 { // Combiners_Opaque_Mod2xNA_Alpha_3s
			mat_diffuse = mesh_color * mix(tex1.rgb * tex2.rgb * 2.0, tex3.rgb, vec3<f32>(tex3.a));
		}
		case 20 { // Combiners_Opaque_AddAlpha_Wgt
			mat_diffuse = mesh_color * tex1.rgb;
			specular = tex2.rgb * tex2.a * u.tex_sample_alpha.y;
		}
		case 21 { // Combiners_Mod_Add_Alpha
			mat_diffuse = mesh_color * tex1.rgb;
			discard_alpha = tex1.a + tex2.a;
			can_discard = true;
			specular = tex2.rgb * (1.0 - tex1.a);
		}
		case 22 { // Combiners_Opaque_ModNA_Alpha
			mat_diffuse = mesh_color * mix(tex1.rgb * tex2.rgb, tex1.rgb, vec3<f32>(tex1.a));
		}
		case 23 { // Combiners_Mod_AddAlpha_Wgt
			mat_diffuse = mesh_color * tex1.rgb;
			discard_alpha = tex1.a;
			can_discard = true;
			specular = tex2.rgb * tex2.a * u.tex_sample_alpha.y;
		}
		case 24 { // Combiners_Opaque_Mod_Add_Wgt
			mat_diffuse = mesh_color * mix(tex1.rgb, tex2.rgb, vec3<f32>(tex2.a));
			specular = tex1.rgb * tex1.a * u.tex_sample_alpha.x;
		}
		case 25 { // Combiners_Opaque_Mod2xNA_Alpha_UnshAlpha
			let glow_opacity = clamp(tex3.a * u.tex_sample_alpha.z, 0.0, 1.0);
			mat_diffuse = mesh_color * mix(tex1.rgb * tex2.rgb * 2.0, tex1.rgb, vec3<f32>(tex1.a)) * (1.0 - glow_opacity);
			specular = tex3.rgb * glow_opacity;
		}
		case 26 { // Combiners_Mod_Dual_Crossfade
			let mixed = mix(mix(tex1, tex2, vec4<f32>(clamp(u.tex_sample_alpha.y, 0.0, 1.0))), tex3, vec4<f32>(clamp(u.tex_sample_alpha.z, 0.0, 1.0)));
			mat_diffuse = mesh_color * mixed.rgb;
			discard_alpha = mixed.a;
			can_discard = true;
		}
		case 27 { // Combiners_Opaque_Mod2xNA_Alpha_Alpha
			mat_diffuse = mesh_color * mix(mix(tex1.rgb * tex2.rgb * 2.0, tex3.rgb, vec3<f32>(tex3.a)), tex1.rgb, vec3<f32>(tex1.a));
		}
		case 28 { // Combiners_Mod_Masked_Dual_Crossfade
			let mixed2 = mix(mix(tex1, tex2, vec4<f32>(clamp(u.tex_sample_alpha.y, 0.0, 1.0))), tex3, vec4<f32>(clamp(u.tex_sample_alpha.z, 0.0, 1.0)));
			mat_diffuse = mesh_color * mixed2.rgb;
			discard_alpha = mixed2.a * tex4.a;
			can_discard = true;
		}
		case 29 { // Combiners_Opaque_Alpha
			mat_diffuse = mesh_color * mix(tex1.rgb, tex2.rgb, vec3<f32>(tex2.a));
		}
		case 30 { // Guild
			mat_diffuse = mesh_color * mix(tex1.rgb * mix(vec3<f32>(1.0), tex2.rgb, vec3<f32>(tex2.a)), tex3.rgb, vec3<f32>(tex3.a));
			discard_alpha = tex1.a;
			can_discard = true;
		}
		case 31 { // Guild_NoBorder
			mat_diffuse = mesh_color * tex1.rgb * mix(vec3<f32>(1.0), tex2.rgb, vec3<f32>(tex2.a));
			discard_alpha = tex1.a;
			can_discard = true;
		}
		case 32 { // Guild_Opaque
			mat_diffuse = mesh_color * mix(tex1.rgb * mix(vec3<f32>(1.0), tex2.rgb, vec3<f32>(tex2.a)), tex3.rgb, vec3<f32>(tex3.a));
		}
		case 33 { // Combiners_Mod_Depth
			mat_diffuse = mesh_color * tex1.rgb;
			discard_alpha = tex1.a;
			can_discard = true;
		}
		case 34 { // Illum
			discard_alpha = tex1.a;
			can_discard = true;
		}
		case 35 { // Combiners_Mod_Mod_Mod_Const
			let combined = tex1 * tex2 * tex3;
			mat_diffuse = mesh_color * combined.rgb;
			discard_alpha = combined.a;
			can_discard = true;
		}
		case 36 { // Combiners_Mod_Mod_Depth
			mat_diffuse = mesh_color * tex1.rgb * tex2.rgb;
			discard_alpha = tex1.a * tex2.a;
			can_discard = true;
		}
		default {
			mat_diffuse = mesh_color * tex1.rgb;
		}
	}

	// final opacity based on blend mode
	var final_opacity: f32;

	if (u.blend_mode == 13) {
		final_opacity = discard_alpha * mesh_opacity;
	} else if (u.blend_mode == 1) {
		final_opacity = mesh_opacity;
		if (can_discard && discard_alpha < u.alpha_test) {
			discard;
		}
	} else if (u.blend_mode == 0) {
		final_opacity = mesh_opacity;
	} else if (u.blend_mode == 4 || u.blend_mode == 5) {
		final_opacity = discard_alpha * mesh_opacity;
		if (can_discard && discard_alpha < u.alpha_test) {
			discard;
		}
	} else {
		final_opacity = discard_alpha * mesh_opacity;
	}

	let lit_color = calc_lighting(mat_diffuse, in.normal);
	return vec4<f32>(lit_color, final_opacity);
}
