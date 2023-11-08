#include "include/cuda_pt_renderer.cuh"
#include "utils.h"

#define CHECK_CUDA_ERROR(func_call) do { \
	auto err = func_call; \
	if (err != cudaSuccess) return TL_ERROR(reinterpret_cast<char const*>(cudaGetErrorString(err))); \
} while (0)


constexpr float k_quad_data_pos_uv[] = {
    // First triangle (positions)
    -1.0f, -1.0f, 0.0f,
    1.0f, -1.0f, 0.0f,
    1.0f,  1.0f, 0.0f,
    // Second triangle (positions)
    -1.0f, -1.0f, 0.0f,
    1.0f,  1.0f, 0.0f,
    -1.0f,  1.0f, 0.0f,
    // First triangle (UVs)
    0.0f, 0.0f,
    1.0f, 0.0f,
    1.0f, 1.0f,
    // Second triangle (UVs)
    0.0f, 0.0f,
    1.0f, 1.0f,
    0.0f, 1.0f
};

static surface<void, cudaSurfaceType2D> g_surf;

GLOBAL void kern_pathtrace(int width, int height) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    glm::vec3 color = glm::vec3(x / (float)width, y / (float)height, 0.0f);
    surf2Dwrite(color, g_surf, x * sizeof(glm::vec3), y);    
}

extern "C"
void launch_kernel(cudaArray *cuda_image_array, dim3 texture_dim) {
	dim3 block_dim(8, 8, 8);
	dim3 grid_dim(texture_dim.x/block_dim.x, texture_dim.y/block_dim.y, texture_dim.z/block_dim.z);
 
    cudaError_t err = cudaBindSurfaceToArray(g_surf, cuda_image_array);
	if( err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
        return;
    }

	kern_pathtrace KERN_PARAMS(grid_dim, block_dim)(texture_dim.x, texture_dim.y);
	
    err = cudaGetLastError();
    if(err != cudaSuccess) {
        std::cerr << "ERROR: " << cudaGetErrorString(err) << std::endl;
    }
}


PTS::CUDA_PTRenderer::CUDA_PTRenderer(RenderConfig config) noexcept : Renderer{config, "CUDA_PTRenderer"} {
	
}

PTS::CUDA_PTRenderer::~CUDA_PTRenderer() noexcept {
    cudaGraphicsUnregisterResource(m_cuda_image_res);
    cudaDeviceReset();
}

auto PTS::CUDA_PTRenderer::init(ObserverPtr<Application> app) noexcept -> tl::expected<void, std::string> {
	TL_CHECK_AND_PASS(
        Renderer::init(app)
    );
    TL_TRY_ASSIGN(
        m_render_tex, GLTexture::create(m_config.width, m_config.height, GL_RGBA)
    );
    CHECK_CUDA_ERROR(
        cudaGraphicsGLRegisterImage(&m_cuda_image_res, m_render_tex->handle(), GL_TEXTURE_2D, cudaGraphicsRegisterFlagsSurfaceLoadStore)
    );
	return {};
}

auto PTS::CUDA_PTRenderer::draw_imgui() noexcept -> tl::expected<void, std::string> {
	return Renderer::draw_imgui();
}

auto PTS::CUDA_PTRenderer::open_scene(View<Scene> scene) noexcept -> tl::expected<void, std::string> {}

auto PTS::CUDA_PTRenderer::on_change_render_config() noexcept -> tl::expected<void, std::string> {
	TL_CHECK_AND_PASS(m_render_tex->bind());
    {
        TL_CHECK_AND_PASS(m_render_tex->resize(config.width, config.height));
    }
    m_render_tex->unbind();
    m_valid = true;
	return {};
}

auto PTS::CUDA_PTRenderer::on_add_editable(EditableView editable) noexcept -> tl::expected<void, std::string> {
    return {};
}

auto PTS::CUDA_PTRenderer::on_remove_editable(EditableView editable) noexcept -> tl::expected<void, std::string> {
    return {};
}

auto PTS::CUDA_PTRenderer::render(View<Camera> camera) noexcept -> tl::expected<void, std::string> {
    return {};
}

auto PTS::CUDA_PTRenderer::render_buffered(View<Camera> camera) noexcept -> tl::expected<TextureHandle, std::string> {
    if(!m_valid) {
        return TL_ERROR("renderer is not valid");
    }

    CHECK_CUDA_ERROR(cudaGraphicsMapResources(1, &m_cuda_image_res));
    {
        cudaArray* dev_img_array = nullptr;
        CHECK_CUDA_ERROR(cudaGraphicsSubResourceGetMappedArray(&dev_img_array, m_cuda_image_res, 0, 0));
        launch_kernel(dev_img_array, dim3(m_config.width, m_config.height, 1));
    }
    CHECK_CUDA_ERROR(cudaGraphicsUnmapResources(1, &m_cuda_image_res));

    return m_render_tex.get();
}

auto PTS::CUDA_PTRenderer::valid() const noexcept -> bool {
    return m_valid;
}
