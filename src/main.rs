use std::{f32::consts::PI, iter};
use wgpu::{util::DeviceExt, RenderPass, TextureUsages};
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Window, WindowBuilder},
};

struct Camera {
    eye: cgmath::Point3<f32>,
    target: cgmath::Point3<f32>,
    up: cgmath::Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
}

#[rustfmt::skip]
pub const OPENGL_TO_WGPU_MATRIX: cgmath::Matrix4<f32> = cgmath::Matrix4::new(
    1.0, 0.0, 0.0, 0.0,
    0.0, 1.0, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.5,
    0.0, 0.0, 0.0, 1.0,
);

impl Camera {
    fn build_view_projection_matrix(&self) -> cgmath::Matrix4<f32> {
        // 1.
        let view = cgmath::Matrix4::look_at_rh(self.eye, self.target, self.up);
        // 2.
        let proj = cgmath::perspective(cgmath::Deg(self.fovy), self.aspect, self.znear, self.zfar);

        // 3.
        return OPENGL_TO_WGPU_MATRIX * proj * view;
    }
}

// We need this for Rust to store our data correctly for the shaders
#[repr(C)]
// This is so we can store this in a buffer
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    // We can't use cgmath with bytemuck directly, so we'll have
    // to convert the Matrix4 into a 4x4 f32 array
    view_proj: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        use cgmath::SquareMatrix;
        Self {
            view_proj: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &Camera) {
        self.view_proj = camera.build_view_projection_matrix().into();
    }
}

const SAMPLE_COUNT: u32 = 4;

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    pos: [f32; 3],
    normal: [f32; 3],
}

impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

fn push_triangle(
    vertices: &mut Vec<Vertex>,
    indices: &mut Vec<u16>,
    v1: usize,
    v2: usize,
    v3: usize,
) {
    let u = [
        vertices[v2].pos[0] - vertices[v1].pos[0],
        vertices[v2].pos[1] - vertices[v1].pos[1],
        vertices[v2].pos[2] - vertices[v1].pos[2],
    ];
    let v = [
        vertices[v3].pos[0] - vertices[v1].pos[0],
        vertices[v3].pos[1] - vertices[v1].pos[1],
        vertices[v3].pos[2] - vertices[v1].pos[2],
    ];
    let normal = [
        u[1] * v[2] - u[2] * v[1], // x-component
        u[2] * v[0] - u[0] * v[2], // y-component
        u[0] * v[1] - u[1] * v[0], // z-component
    ];
    vertices[v1].normal[0] += normal[0];
    vertices[v1].normal[1] += normal[1];
    vertices[v1].normal[2] += normal[2];
    vertices[v2].normal[0] += normal[0];
    vertices[v2].normal[1] += normal[1];
    vertices[v2].normal[2] += normal[2];
    vertices[v3].normal[0] += normal[0];
    vertices[v3].normal[1] += normal[1];
    vertices[v3].normal[2] += normal[2];

    indices.push(v1 as u16);
    indices.push(v2 as u16);
    indices.push(v3 as u16);
}

fn generate_sphere(radius: f32, pos: [f32; 3]) -> (Vec<Vertex>, Vec<u16>) {
    const LEN: usize = 30;
    const CIRCLE_POINTS: usize = 30;
    let mut vertices = vec![];

    for ring in 0..LEN {
        let v = ring as f32 / LEN as f32;
        let phi = v * PI;

        for segment in 0..CIRCLE_POINTS {
            let u = segment as f32 / CIRCLE_POINTS as f32;
            let theta = u * 2.0 * PI;

            let x = theta.sin() * phi.sin();
            let y = phi.cos();
            let z = theta.cos() * phi.sin();

            let pos = [x, y, z];

            vertices.push(Vertex {
                pos,
                normal: [0.0, 0.0, 0.0],
            });
        }
    }

    let mut indices = Vec::new();

    for i in 0..LEN - 1 {
        // Push last triangle
        push_triangle(
            &mut vertices,
            &mut indices,
            (i + 1) * CIRCLE_POINTS - 1,
            (i + 2) * CIRCLE_POINTS - 1,
            i * CIRCLE_POINTS,
        );
        push_triangle(
            &mut vertices,
            &mut indices,
            (i + 2) * CIRCLE_POINTS - 1,
            (i + 1) * CIRCLE_POINTS,
            i * CIRCLE_POINTS,
        );

        for j in 0..CIRCLE_POINTS - 1 {
            push_triangle(
                &mut vertices,
                &mut indices,
                i * CIRCLE_POINTS + j,
                (i + 1) * CIRCLE_POINTS + j,
                i * CIRCLE_POINTS + j + 1,
            );
            push_triangle(
                &mut vertices,
                &mut indices,
                (i + 1) * CIRCLE_POINTS + j,
                (i + 1) * CIRCLE_POINTS + j + 1,
                i * CIRCLE_POINTS + j + 1,
            );
        }
    }

    for vertex in &mut vertices {
        let multitude =
            1.0 / (vertex.normal[0].powi(2) + vertex.normal[1].powi(2) + vertex.normal[2].powi(2));
        vertex.normal[0] *= multitude;
        vertex.normal[1] *= multitude;
        vertex.normal[2] *= multitude;

        vertex.pos[0] = vertex.pos[0] * radius + pos[0];
        vertex.pos[1] = vertex.pos[1] * radius + pos[1];
        vertex.pos[2] = vertex.pos[2] * radius + pos[2];
    }

    (vertices, indices)
}

fn generate_snake() -> (Vec<Vertex>, Vec<u16>) {
    const LEN: usize = 200;
    const CIRCLE_POINTS: usize = 30;
    let mut vertices = vec![
        Vertex {
            pos: [0.0, 0.0, 0.0],
            normal: [0.0, 1.0, 0.0]
        };
        CIRCLE_POINTS * LEN
    ];

    const START_X: f32 = -50.;
    const END_X: f32 = 50.;
    const RADIUS: f32 = 2.;

    for i in 0..LEN {
        let x = (END_X - START_X) * (i as f32) / (LEN as f32) + START_X;
        let y = (x / 3.0).sin() * 5.0;

        // Compute the tangent vector (derivative of the curve)
        let tangent_x = 1.0; // dx/dt = 1 (since x = t)
        let tangent_y = (x / 3.0).cos() * (5.0 / 3.0); // dy/dx = cos(x/3) * (2/3)
        let tangent_z = 0.0; // dz/dt = 0

        // Normalize the tangent vector
        let tangent_length =
            (tangent_x * tangent_x + tangent_y * tangent_y + tangent_z * tangent_z).sqrt();
        let tangent = [
            tangent_x / tangent_length,
            tangent_y / tangent_length,
            tangent_z / tangent_length,
        ];

        // Compute the normal vector (perpendicular to the tangent in the XY plane)
        let normal = [-tangent[1], tangent[0], 0.0];

        // Compute the binormal vector (perpendicular to both tangent and normal)
        let binormal = [
            tangent[1] * normal[2] - tangent[2] * normal[1],
            tangent[2] * normal[0] - tangent[0] * normal[2],
            tangent[0] * normal[1] - tangent[1] * normal[0],
        ];

        for j in 0..CIRCLE_POINTS {
            let angle = (PI * 2.0) * (j as f32) / (CIRCLE_POINTS as f32);
            let y1 = angle.cos() * RADIUS;
            let z1 = angle.sin() * RADIUS;

            let circle_point = [
                x + normal[0] * y1 + binormal[0] * z1,
                y + normal[1] * y1 + binormal[1] * z1,
                normal[2] * y1 + binormal[2] * z1,
            ];

            vertices[i * CIRCLE_POINTS + j].pos = circle_point;
        }
    }

    let mut indices = Vec::new();

    for i in 0..LEN - 1 {
        // Push last triangle
        push_triangle(
            &mut vertices,
            &mut indices,
            (i + 1) * CIRCLE_POINTS - 1,
            i * CIRCLE_POINTS,
            (i + 2) * CIRCLE_POINTS - 1,
        );
        push_triangle(
            &mut vertices,
            &mut indices,
            (i + 2) * CIRCLE_POINTS - 1,
            i * CIRCLE_POINTS,
            (i + 1) * CIRCLE_POINTS,
        );

        for j in 0..CIRCLE_POINTS - 1 {
            push_triangle(
                &mut vertices,
                &mut indices,
                i * CIRCLE_POINTS + j,
                i * CIRCLE_POINTS + j + 1,
                (i + 1) * CIRCLE_POINTS + j,
            );
            push_triangle(
                &mut vertices,
                &mut indices,
                (i + 1) * CIRCLE_POINTS + j,
                i * CIRCLE_POINTS + j + 1,
                (i + 1) * CIRCLE_POINTS + j + 1,
            );
        }
    }

    for vertex in &mut vertices {
        let multitude =
            1.0 / (vertex.normal[0].powi(2) + vertex.normal[1].powi(2) + vertex.normal[2].powi(2));
        vertex.normal[0] *= multitude;
        vertex.normal[1] *= multitude;
        vertex.normal[2] *= multitude;
    }

    (vertices, indices)
}

struct State<'a> {
    surface: wgpu::Surface<'a>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: winit::dpi::PhysicalSize<u32>,
    render_pipeline: wgpu::RenderPipeline,
    multisample_texture: Option<wgpu::Texture>,

    snake: Snake,
    sphere: Sphere,

    camera: Camera,
    camera_uniform: CameraUniform,
    camera_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    camera_controller: CameraController,

    depth_texture: wgpu::Texture,
    depth_texture_view: wgpu::TextureView,

    window: &'a Window,
}

struct Snake {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl Snake {
    fn new(device: &wgpu::Device) -> Self {
        let (mut vertices, mut indices) = generate_snake();
        indices.push(0);

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = indices.len() as u32;
        Self {
            vertex_buffer,
            index_buffer,
            num_indices,
        }
    }

    fn render(&self, render_pass: &mut RenderPass) {
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}

struct Sphere {
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    num_indices: u32,
}

impl Sphere {
    fn new(device: &wgpu::Device, radius: f32, center: [f32; 3]) -> Self {
        let (vertices, indices) = generate_sphere(radius, center);
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Vertex Buffer"),
            contents: bytemuck::cast_slice(vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Index Buffer"),
            contents: bytemuck::cast_slice(indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });
        let num_indices = indices.len() as u32;
        Self {
            vertex_buffer,
            index_buffer,
            num_indices,
        }
    }

    fn render(&self, render_pass: &mut wgpu::RenderPass) {
        render_pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
        render_pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint16);
        render_pass.draw_indexed(0..self.num_indices, 0, 0..1);
    }
}

impl<'a> State<'a> {
    async fn new(window: &'a Window) -> State<'a> {
        let size = window.inner_size();

        // The instance is a handle to our GPU
        // BackendBit::PRIMARY => Vulkan + Metal + DX12 + Browser WebGPU
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::all(),
            ..Default::default()
        });

        // # Safety
        //
        // The surface needs to live as long as the window that created it.
        // State owns the window so this should be safe.
        let surface = instance.create_surface(window).unwrap();

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .unwrap();

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,
                    required_features: wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES,
                    required_limits: wgpu::Limits::default(),
                    memory_hints: Default::default(),
                },
                None, // Trace path
            )
            .await
            .unwrap();

        let surface_caps = surface.get_capabilities(&adapter);
        // Shader code in this tutorial assumes an Srgb surface texture. Using a different
        // one will result all the colors comming out darker. If you want to support non
        // Srgb surfaces, you'll need to account for that when drawing to the frame.
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shader.wgsl").into()),
        });

        let camera = Camera {
            // position the camera 1 unit up and 2 units back
            // +z is out of the screen
            eye: (0.0, 1.0, 2.0).into(),
            // have it look at the origin
            target: (0.0, 0.0, 0.0).into(),
            // which way is "up"
            up: cgmath::Vector3::unit_y(),
            aspect: config.width as f32 / config.height as f32,
            fovy: 45.0,
            znear: 0.1,
            zfar: 100.0,
        };

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Camera Buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("camera_bind_group_layout"),
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &camera_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: camera_buffer.as_entire_binding(),
            }],
            label: Some("camera_bind_group"),
        });

        let render_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Render Pipeline Layout"),
                bind_group_layouts: &[&camera_bind_group_layout],
                push_constant_ranges: &[],
            });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Render Pipeline"),
            layout: Some(&render_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[Vertex::desc()],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState {
                        color: wgpu::BlendComponent::REPLACE,
                        alpha: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::POLYGON_MODE_LINE
                // or Features::POLYGON_MODE_POINT
                polygon_mode: wgpu::PolygonMode::Fill,
                // Requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // Requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: wgpu::TextureFormat::Depth32Float,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::LessEqual,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: SAMPLE_COUNT,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            // If the pipeline will be used with a multiview render pass, this
            // indicates how many array layers the attachments will have.
            multiview: None,
            // Useful for optimizing shader compilation on Android
            cache: None,
        });

        let depth_texture = device.create_texture(&wgpu::TextureDescriptor {
            size: wgpu::Extent3d {
                width: config.width,
                height: config.height,
                depth_or_array_layers: 1,
            },
            view_formats: &[wgpu::TextureFormat::Depth32Float],
            mip_level_count: 1,
            sample_count: SAMPLE_COUNT,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float, // Common depth format
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            label: Some("depth_texture"),
        });

        let depth_texture_view = depth_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let camera_controller = CameraController::new(0.3);

        let snake = Snake::new(&device);

        let sphere = Sphere::new(&device, 5.0, [-50.0, (-50.0f32 / 3.0).sin() * 5.0, 0.0]);

        Self {
            surface,
            device,
            queue,
            config,
            size,
            render_pipeline,
            snake,
            sphere,

            multisample_texture: None,

            camera,
            camera_uniform,
            camera_buffer,
            camera_bind_group,
            camera_controller,

            depth_texture,
            depth_texture_view,

            window,
        }
    }

    pub fn window(&self) -> &Window {
        &self.window
    }

    pub fn resize(&mut self, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn input(&mut self, event: &WindowEvent) -> bool {
        self.camera_controller.process_events(event)
    }

    fn update(&mut self) {
        self.camera_controller.update_camera(&mut self.camera);
        self.camera_uniform.update_view_proj(&self.camera);
        self.queue.write_buffer(
            &self.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.camera_uniform]),
        );
    }

    fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        let State {
            device,
            queue,
            surface,
            ..
        } = self;

        let output = surface.get_current_texture()?;
        let output_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let multisample_texture = match &mut self.multisample_texture {
            None => {
                self.multisample_texture = Some(device.create_texture(&wgpu::TextureDescriptor {
                    label: Some("multisample texture"),
                    size: wgpu::Extent3d {
                        width: self.config.width,
                        height: self.config.height,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: SAMPLE_COUNT,
                    dimension: wgpu::TextureDimension::D2,
                    format: self.config.format,
                    usage: TextureUsages::RENDER_ATTACHMENT,
                    view_formats: &[self.config.format],
                }));
                self.multisample_texture.as_mut().unwrap()
            }
            Some(texture) => {
                if texture.size().width != self.config.width
                    || texture.size().height != self.config.height
                {
                    *texture = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("multisample texture"),
                        size: wgpu::Extent3d {
                            width: self.config.width,
                            height: self.config.height,
                            depth_or_array_layers: 1,
                        },
                        mip_level_count: 1,
                        sample_count: SAMPLE_COUNT,
                        dimension: wgpu::TextureDimension::D2,
                        format: self.config.format,
                        usage: TextureUsages::RENDER_ATTACHMENT,
                        view_formats: &[self.config.format],
                    });
                }
                texture
            }
        };
        let view = multisample_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Render Encoder"),
        });

        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: Some(&output_view),
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.0,
                            g: 0.1,
                            b: 0.2,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],

                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0), // Clear depth buffer to 1.0 (far plane)
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, &self.camera_bind_group, &[]);
            self.snake.render(&mut render_pass);
            self.sphere.render(&mut render_pass);
        }

        queue.submit(iter::once(encoder.finish()));
        output.present();

        Ok(())
    }
}

struct CameraController {
    speed: f32,
    is_forward_pressed: bool,
    is_backward_pressed: bool,
    is_left_pressed: bool,
    is_right_pressed: bool,
}

impl CameraController {
    fn new(speed: f32) -> Self {
        Self {
            speed,
            is_forward_pressed: false,
            is_backward_pressed: false,
            is_left_pressed: false,
            is_right_pressed: false,
        }
    }

    fn process_events(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state,
                        physical_key: PhysicalKey::Code(keycode),
                        ..
                    },
                ..
            } => {
                let is_pressed = *state == ElementState::Pressed;
                match keycode {
                    KeyCode::KeyW | KeyCode::ArrowUp => {
                        self.is_forward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyA | KeyCode::ArrowLeft => {
                        self.is_left_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyS | KeyCode::ArrowDown => {
                        self.is_backward_pressed = is_pressed;
                        true
                    }
                    KeyCode::KeyD | KeyCode::ArrowRight => {
                        self.is_right_pressed = is_pressed;
                        true
                    }
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn update_camera(&self, camera: &mut Camera) {
        use cgmath::InnerSpace;
        let forward = camera.target - camera.eye;
        let forward_norm = forward.normalize();
        let forward_mag = forward.magnitude();

        // Prevents glitching when the camera gets too close to the
        // center of the scene.
        if self.is_forward_pressed && forward_mag > self.speed {
            camera.eye += forward_norm * self.speed;
        }
        if self.is_backward_pressed {
            camera.eye -= forward_norm * self.speed;
        }

        let right = forward_norm.cross(camera.up);

        // Redo radius calc in case the forward/backward is pressed.
        let forward = camera.target - camera.eye;
        let forward_mag = forward.magnitude();

        if self.is_right_pressed {
            // Rescale the distance between the target and the eye so
            // that it doesn't change. The eye, therefore, still
            // lies on the circle made by the target and eye.
            camera.eye = camera.target - (forward + right * self.speed).normalize() * forward_mag;
        }
        if self.is_left_pressed {
            camera.eye = camera.target - (forward - right * self.speed).normalize() * forward_mag;
        }
    }
}

pub async fn run() {
    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    // State::new uses async code, so we're going to wait for it to finish
    let mut state = State::new(&window).await;
    let mut surface_configured = false;

    event_loop
        .run(move |event, control_flow| {
            match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == state.window().id() => {
                    if !state.input(event) {
                        match event {
                            WindowEvent::CloseRequested
                            | WindowEvent::KeyboardInput {
                                event:
                                    KeyEvent {
                                        state: ElementState::Pressed,
                                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                                        ..
                                    },
                                ..
                            } => control_flow.exit(),
                            WindowEvent::Resized(physical_size) => {
                                surface_configured = true;
                                state.resize(*physical_size);
                            }
                            WindowEvent::RedrawRequested => {
                                // This tells winit that we want another frame after this one
                                state.window().request_redraw();

                                if !surface_configured {
                                    return;
                                }

                                state.update();
                                match state.render() {
                                    Ok(_) => {}
                                    // Reconfigure the surface if it's lost or outdated
                                    Err(
                                        wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated,
                                    ) => state.resize(state.size),
                                    // The system is out of memory, we should probably quit
                                    Err(
                                        wgpu::SurfaceError::OutOfMemory | wgpu::SurfaceError::Other,
                                    ) => {
                                        log::error!("OutOfMemory");
                                        control_flow.exit();
                                    }

                                    // This happens when the a frame takes too long to present
                                    Err(wgpu::SurfaceError::Timeout) => {
                                        log::warn!("Surface timeout")
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                _ => {}
            }
        })
        .unwrap();
}

fn main() {
    pollster::block_on(run())
}
