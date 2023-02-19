#![feature(get_many_mut)]
//! A shader that renders a mesh multiple times in one draw call.

use std::{collections::HashMap, time::Instant};

use bevy::{
    core_pipeline::core_3d::Transparent3d,
    diagnostic::{Diagnostic, DiagnosticId, Diagnostics},
    ecs::{
        query::QueryItem,
        system::{lifetimeless::*, SystemParamItem},
    },
    pbr::{MeshPipeline, MeshPipelineKey, MeshUniform, SetMeshBindGroup, SetMeshViewBindGroup},
    prelude::*,
    render::{
        extract_component::{ExtractComponent, ExtractComponentPlugin},
        mesh::{GpuBufferInfo, MeshVertexBufferLayout},
        render_asset::RenderAssets,
        render_phase::{
            AddRenderCommand, DrawFunctions, EntityRenderCommand, RenderCommandResult, RenderPhase,
            SetItemPipeline, TrackedRenderPass,
        },
        render_resource::*,
        renderer::RenderDevice,
        view::{ExtractedView, NoFrustumCulling},
        RenderApp, RenderStage,
    },
};
use bytemuck::{Pod, Zeroable};

const N: i32 = 20;
const RADIUS: f32 = 1.0 / 3.0 / N as f32 * 10.;
const GRAVITY: f32 = -0.00981;
const PHYSICS_SUB_STEPS: i32 = 8;
const FPS: i32 = 99;

fn solve_collision(i1: &mut InstanceData, i2: &mut InstanceData) {
    let d2 = i1.position.distance_squared(i2.position);
    if d2 < 4. * RADIUS * RADIUS {
        let d = d2.sqrt();
        let overlap = 2. * RADIUS - d;
        let shift = overlap / 2.;

        let p1 = &mut i1.position;
        let p2 = &mut i2.position;

        let p12 = *p1 - *p2;
        let p21 = *p2 - *p1;
        *p1 += shift * p12;
        *p2 += shift * p21;

        // No idea how the fuck to derive this
        // But it's the collision handling from
        // https://physics.stackexchange.com/a/681574
        let p21 = *p2 - *p1;
        let n = p21.normalize();

        let meff = 0.5f32;
        let vimp = n.dot(i1.velocity - i2.velocity);
        let eps = 0.5;
        let j = (1. + eps) * meff * vimp;

        let dv1 = -j * n;
        let dv2 = j * n;

        i1.velocity += dv1;
        i2.velocity += dv2;
    }
}

const EXTENT: f32 = 10.;

const PHYSICS_STEPS_MS: DiagnosticId =
    DiagnosticId::from_u128(305697860378959379917727562126468611595);

fn physics_step(mut q: Query<&mut InstanceMaterialData>, mut diagnostics: ResMut<Diagnostics>) {
    let tic = Instant::now();
    for mut instance_material_data in q.iter_mut() {
        for _ in 0..PHYSICS_SUB_STEPS {
            // let original = instance_material_data.clone();
            for i in 0..instance_material_data.0.len() {
                let instance = &mut instance_material_data.0[i];
                instance.velocity += Vec3::new(0., GRAVITY / (PHYSICS_SUB_STEPS * FPS) as f32, 0.);

                if instance.position.x > EXTENT {
                    instance.position.x = EXTENT;
                    instance.velocity.x *= -0.9;
                }
                if instance.position.x < -EXTENT {
                    instance.position.x = -EXTENT;
                    instance.velocity.x *= -0.9;
                }
                if instance.position.y > EXTENT {
                    instance.position.y = EXTENT;
                    instance.velocity.y *= -0.9;
                }
                if instance.position.y < -EXTENT {
                    instance.position.y = -EXTENT;
                    instance.velocity.y *= -0.9;
                }
                instance.position += instance.velocity;

                for j in instance_material_data.1.neighbors(instance_material_data.0[i].position) {
                    if i != j {
                        let [a, b] = instance_material_data.0.get_many_mut([i, j]).unwrap();
                        solve_collision(a, b);
                    }
                }
            }
        }
        instance_material_data.1 = SpatialHashGrid::from_instance_data(&instance_material_data.0);
    }
    let ms = tic.elapsed().as_secs_f32() * 1000.;
    diagnostics.add_measurement(PHYSICS_STEPS_MS, || ms as f64);
}

use bevy::diagnostic::{FrameTimeDiagnosticsPlugin, LogDiagnosticsPlugin};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_plugin(LogDiagnosticsPlugin::default())
        .add_plugin(FrameTimeDiagnosticsPlugin::default())
        .add_plugin(CustomMaterialPlugin)
        .add_system(physics_step)
        .add_startup_system(setup)
        .run();
}

fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut diagnostics: ResMut<Diagnostics>,
) {
    diagnostics.add(Diagnostic::new(PHYSICS_STEPS_MS, "physics_step_ms", 20).with_suffix("ms"));
    commands.spawn((
        meshes.add(Mesh::from(shape::UVSphere {
            radius: RADIUS,
            sectors: 18,
            stacks: 18,
        })),
        SpatialBundle::VISIBLE_IDENTITY,
        InstanceMaterialData::from_instance_data(
            (1..=N)
                .flat_map(|x| (1..=N).map(move |y| (x as f32 / N as f32, y as f32 / N as f32)))
                .map(|(x, y)| InstanceData {
                    position: Vec3::new(x * 10.0 - 5.0, y * 10.0 - 5.0, 0.0),
                    velocity: Vec3::new(0.01, 0., 0.),
                    scale: 1.0,
                    color: Color::hsla(x * 360., y, 0.5, 1.0).as_rgba_f32(),
                })
                .collect(),
        ),
        // NOTE: Frustum culling is done based on the Aabb of the Mesh and the GlobalTransform.
        // As the cube is at the origin, if its Aabb moves outside the view frustum, all the
        // instanced cubes will be culled.
        // The InstanceMaterialData contains the 'GlobalTransform' information for this custom
        // instancing, and that is not taken into account with the built-in frustum culling.
        // We must disable the built-in frustum culling by adding the `NoFrustumCulling` marker
        // component to avoid incorrect culling.
        NoFrustumCulling,
    ));

    // camera
    commands.spawn(Camera3dBundle {
        transform: Transform::from_xyz(0.0, 0.0, 15.0).looking_at(Vec3::ZERO, Vec3::Y),
        projection: OrthographicProjection {
            scale: 3.0,
            scaling_mode: bevy::render::camera::ScalingMode::Auto {
                min_width: 1920. / 128.,
                min_height: 1080. / 128.,
            },
            ..default()
        }
        .into(),
        ..default()
    });
}

#[derive(Clone)]
pub struct SpatialHashGrid {
    cells: HashMap<[i32; 3], Vec<usize>>,
}

impl SpatialHashGrid {
    pub fn new() -> Self {
        Self {
            cells: HashMap::new(),
        }
    }

    pub fn from_instance_data(instance_data: &Vec<InstanceData>) -> Self {
        let mut grid = Self::new();
        for (i, instance) in instance_data.iter().enumerate() {
            grid.insert(instance.position, i);
        }
        grid
    }

    pub fn insert(&mut self, position: Vec3, index: usize) {
        let cell = self.cell(position);
        self.cells.entry(cell).or_default().push(index);
    }

    pub fn remove(&mut self, position: Vec3, index: usize) {
        let cell = self.cell(position);
        if let Some(indices) = self.cells.get_mut(&cell) {
            indices.retain(|&i| i != index);
        }
    }

    pub fn cell(&self, position: Vec3) -> [i32; 3] {
        [
            (position.x / RADIUS).floor() as i32,
            (position.y / RADIUS).floor() as i32,
            (position.z / RADIUS).floor() as i32,
        ]
    }

    pub fn neighbors(&self, position: Vec3) -> Vec<usize> {
        let cell = self.cell(position);
        let mut neighbors = Vec::new();
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if let Some(indices) = self.cells.get(&[cell[0] + x, cell[1] + y, cell[2] + z])
                    {
                        neighbors.extend_from_slice(indices);
                    }
                }
            }
        }
        neighbors
    }
}

#[derive(Component)]
struct InstanceMaterialData(Vec<InstanceData>, SpatialHashGrid);

impl InstanceMaterialData {
    fn from_instance_data(instance_data: Vec<InstanceData>) -> Self {
        
        let grid = SpatialHashGrid::from_instance_data(&instance_data);
        Self(instance_data, grid)
    }
}

impl ExtractComponent for InstanceMaterialData {
    type Query = &'static InstanceMaterialData;
    type Filter = ();

    fn extract_component(item: QueryItem<'_, Self::Query>) -> Self {
        InstanceMaterialData(item.0.clone(), item.1.clone())
    }
}

pub struct CustomMaterialPlugin;

impl Plugin for CustomMaterialPlugin {
    fn build(&self, app: &mut App) {
        app.add_plugin(ExtractComponentPlugin::<InstanceMaterialData>::default());
        app.sub_app_mut(RenderApp)
            .add_render_command::<Transparent3d, DrawCustom>()
            .init_resource::<CustomPipeline>()
            .init_resource::<SpecializedMeshPipelines<CustomPipeline>>()
            .add_system_to_stage(RenderStage::Queue, queue_custom)
            .add_system_to_stage(RenderStage::Prepare, prepare_instance_buffers);
    }
}

#[derive(Clone, Copy, Pod, Zeroable)]
#[repr(C)]
pub struct InstanceData {
    velocity: Vec3,
    position: Vec3,
    scale: f32,
    color: [f32; 4],
}

#[allow(clippy::too_many_arguments)]
fn queue_custom(
    transparent_3d_draw_functions: Res<DrawFunctions<Transparent3d>>,
    custom_pipeline: Res<CustomPipeline>,
    msaa: Res<Msaa>,
    mut pipelines: ResMut<SpecializedMeshPipelines<CustomPipeline>>,
    mut pipeline_cache: ResMut<PipelineCache>,
    meshes: Res<RenderAssets<Mesh>>,
    material_meshes: Query<(Entity, &MeshUniform, &Handle<Mesh>), With<InstanceMaterialData>>,
    mut views: Query<(&ExtractedView, &mut RenderPhase<Transparent3d>)>,
) {
    let draw_custom = transparent_3d_draw_functions
        .read()
        .get_id::<DrawCustom>()
        .unwrap();

    let msaa_key = MeshPipelineKey::from_msaa_samples(msaa.samples);

    for (view, mut transparent_phase) in &mut views {
        let view_key = msaa_key | MeshPipelineKey::from_hdr(view.hdr);
        let rangefinder = view.rangefinder3d();
        for (entity, mesh_uniform, mesh_handle) in &material_meshes {
            if let Some(mesh) = meshes.get(mesh_handle) {
                let key =
                    view_key | MeshPipelineKey::from_primitive_topology(mesh.primitive_topology);
                let pipeline = pipelines
                    .specialize(&mut pipeline_cache, &custom_pipeline, key, &mesh.layout)
                    .unwrap();
                transparent_phase.add(Transparent3d {
                    entity,
                    pipeline,
                    draw_function: draw_custom,
                    distance: rangefinder.distance(&mesh_uniform.transform),
                });
            }
        }
    }
}

#[derive(Component)]
pub struct InstanceBuffer {
    buffer: Buffer,
    length: usize,
}

fn prepare_instance_buffers(
    mut commands: Commands,
    query: Query<(Entity, &InstanceMaterialData)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, instance_data) in &query {
        let buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("instance data buffer"),
            contents: bytemuck::cast_slice(instance_data.0.as_slice()),
            usage: BufferUsages::VERTEX | BufferUsages::COPY_DST,
        });
        commands.entity(entity).insert(InstanceBuffer {
            buffer,
            length: instance_data.0.len(),
        });
    }
}

#[derive(Resource)]
pub struct CustomPipeline {
    shader: Handle<Shader>,
    mesh_pipeline: MeshPipeline,
}

impl FromWorld for CustomPipeline {
    fn from_world(world: &mut World) -> Self {
        let asset_server = world.resource::<AssetServer>();
        let shader = asset_server.load("shaders/instancing.wgsl");

        let mesh_pipeline = world.resource::<MeshPipeline>();

        CustomPipeline {
            shader,
            mesh_pipeline: mesh_pipeline.clone(),
        }
    }
}

impl SpecializedMeshPipeline for CustomPipeline {
    type Key = MeshPipelineKey;

    fn specialize(
        &self,
        key: Self::Key,
        layout: &MeshVertexBufferLayout,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh_pipeline.specialize(key, layout)?;
        descriptor.vertex.shader = self.shader.clone();
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as u64,
            step_mode: VertexStepMode::Instance,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32.size() * 3,
                    shader_location: 3, // shader locations 0-2 are taken up by Position, Normal and UV attributes
                },
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: VertexFormat::Float32.size() * (3 + 3 + 1),
                    shader_location: 4,
                },
            ],
        });
        descriptor.fragment.as_mut().unwrap().shader = self.shader.clone();
        descriptor.layout = Some(vec![
            self.mesh_pipeline.view_layout.clone(),
            self.mesh_pipeline.mesh_layout.clone(),
        ]);

        Ok(descriptor)
    }
}

type DrawCustom = (
    SetItemPipeline,
    SetMeshViewBindGroup<0>,
    SetMeshBindGroup<1>,
    DrawMeshInstanced,
);

pub struct DrawMeshInstanced;

impl EntityRenderCommand for DrawMeshInstanced {
    type Param = (
        SRes<RenderAssets<Mesh>>,
        SQuery<Read<Handle<Mesh>>>,
        SQuery<Read<InstanceBuffer>>,
    );
    #[inline]
    fn render<'w>(
        _view: Entity,
        item: Entity,
        (meshes, mesh_query, instance_buffer_query): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let mesh_handle = mesh_query.get(item).unwrap();
        let instance_buffer = instance_buffer_query.get_inner(item).unwrap();

        let gpu_mesh = match meshes.into_inner().get(mesh_handle) {
            Some(gpu_mesh) => gpu_mesh,
            None => return RenderCommandResult::Failure,
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, instance_buffer.buffer.slice(..));

        match &gpu_mesh.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                pass.draw_indexed(0..*count, 0, 0..instance_buffer.length as u32);
            }
            GpuBufferInfo::NonIndexed { vertex_count } => {
                pass.draw(0..*vertex_count, 0..instance_buffer.length as u32);
            }
        }
        RenderCommandResult::Success
    }
}
