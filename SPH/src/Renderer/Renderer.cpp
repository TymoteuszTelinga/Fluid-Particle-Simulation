
#include "Renderer.h"
#include "Texture.h"
#include "VertexArray.h"
#include "IndexBuffer.h"

#include "Core/Base.h"

#include <GL/glew.h>
#include <glm/gtx/quaternion.hpp>
#include <stdio.h>

static void GLAPIENTRY MessageCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam)
{
	if (type == GL_DEBUG_TYPE_ERROR)
	{
		fprintf(stderr, "GL CALLBACK: %s type = 0x%x, severity = 0x%x, message = %s\n",
			(type == GL_DEBUG_TYPE_ERROR ? "** GL ERROR **" : ""),
			type, severity, message);
	}
}

struct QuadVertex
{
	glm::vec2 Position;
	glm::vec2 UV;
	glm::vec3 Color;
};

struct RendererData
{
	static const uint32_t MaxQuads = 20000;
	static const uint32_t MaxVertices = MaxQuads * 4;
	static const uint32_t MaxIndices = MaxQuads * 6;
	float ParticleSize = 5.f;

	//Particle VAO, VBO, IBO
	Ref<VertexArray> QuadVertexArray;
	Ref<VertexBuffer> QuadVertexBuffer;
	Ref<IndexBuffer> QuadIndexBuffer;
	//Obstacles VAO, VBO, IBO
	Ref<VertexArray> BackgroundVertexArray;
	Ref<VertexBuffer> BackgroundBuffer;
	Ref<IndexBuffer> BackgroundIndexBuffer;

	Ref<Shader> QuadShader;
	Ref<Shader> BackgroundShader;

	uint32_t QuadIndexCount = 0;
	QuadVertex* QuadVertexBufferBase = nullptr; // store adres of cpu array
	QuadVertex* QuadVertexBufferPtr = nullptr; // store aders of next Quad in cpu array

	glm::vec4 QuadVertexPositions[4];
};

static RendererData s_Data;

struct CameraData
{
	glm::mat4 ViewProjectionMatrix = glm::mat4(1);
	glm::vec3 CameraPosition = glm::vec3(0.f);
	glm::vec2 BBoxMin = glm::vec2(0);
	glm::vec2 BBoxMax = glm::vec2(0);
	float CameraWidth = 100;
};

static CameraData s_CameraData;

Renderer::Statistics s_Stats;

void Renderer::Init()
{
	//load openGL funcions
	if (glewInit() != GLEW_OK)
		return;

	//setup OpenGl Debug Callback
	glEnable(GL_DEBUG_OUTPUT);
	glDebugMessageCallback(MessageCallback, 0);

	//glEnable(GL_DEPTH_TEST);

	glEnable(GL_BLEND);
	//glBlendEquationSeparate(GL_FUNC_ADD, GL_MAX);
	//glBlendEquation(GL_MAX);
	//glBlendFunc(GL_ONE, GL_ONE);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	//glBlendFunc(GL_ONE_MINUS_DST_ALPHA, GL_DST_ALPHA);

	//Init Data
	s_Data.QuadVertexArray = CreateRef<VertexArray>();
	s_Data.QuadVertexBuffer = CreateRef<VertexBuffer>(nullptr, s_Data.MaxVertices * sizeof(QuadVertex));
	VertexBufferLayout Layout;
	Layout.Pushf(2); //Position
	Layout.Pushf(2); //Texture
	Layout.Pushf(3); //Tint Color
	s_Data.QuadVertexArray->AddBuffer(*s_Data.QuadVertexBuffer, Layout);

	s_Data.QuadVertexBufferBase = new QuadVertex[s_Data.MaxVertices];

	//precalculate Inices
	uint32_t* quadIndices = new uint32_t[s_Data.MaxIndices];
	uint32_t offset = 0;
	for (uint32_t i = 0; i < s_Data.MaxIndices; i += 6)
	{
		quadIndices[i + 0] = offset + 0;
		quadIndices[i + 1] = offset + 1;
		quadIndices[i + 2] = offset + 2;

		quadIndices[i + 3] = offset + 2;
		quadIndices[i + 4] = offset + 3;
		quadIndices[i + 5] = offset + 0;

		offset += 4;
	}
	s_Data.QuadIndexBuffer = CreateRef<IndexBuffer>(quadIndices, s_Data.MaxIndices);
	delete[] quadIndices;

	s_Data.QuadVertexPositions[0] = { -s_Data.ParticleSize, -s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[1] = {  s_Data.ParticleSize, -s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[2] = {  s_Data.ParticleSize,  s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[3] = { -s_Data.ParticleSize,  s_Data.ParticleSize, 0.0f, 1.0f };

	s_Data.QuadShader = CreateRef<Shader>("src\\Shaders\\vertex.glsl", "src\\Shaders\\fragment.glsl");

	//background Data
	s_Data.BackgroundIndexBuffer = CreateRef<IndexBuffer>(nullptr, 0);
	s_Data.BackgroundBuffer = CreateRef<VertexBuffer>(nullptr, 0);

	s_Data.BackgroundVertexArray = CreateRef<VertexArray>();
	s_Data.BackgroundVertexArray->AddBuffer(*s_Data.BackgroundBuffer, Layout);
	s_Data.BackgroundShader = CreateRef<Shader>("src\\Shaders\\vertex.glsl", "src\\Shaders\\fragmentBG.glsl");

}

void Renderer::Shutdown()
{
	delete[] s_Data.QuadVertexBufferBase;
}

void Renderer::SetColor(float r, float g, float b, float a)
{
	glClearColor(r, g, b, a);
}

void Renderer::Resize(int width, int height)
{
	glViewport(0, 0, width, height);
}

void Renderer::Clear()
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
}

void Renderer::SetObstacles(const std::vector<Obstacle>& obstacles)
{
	uint32_t Indices = obstacles.size() * 6;
	uint32_t* quadIndices = new uint32_t[Indices];
	uint32_t offset = 0;
	for (uint32_t i = 0; i < Indices; i += 6)
	{
		quadIndices[i + 0] = offset + 0;
		quadIndices[i + 1] = offset + 1;
		quadIndices[i + 2] = offset + 2;

		quadIndices[i + 3] = offset + 2;
		quadIndices[i + 4] = offset + 3;
		quadIndices[i + 5] = offset + 0;

		offset += 4;
	}
	s_Data.BackgroundIndexBuffer = CreateRef<IndexBuffer>(quadIndices, Indices);
	delete[] quadIndices;

	uint32_t Vertexes = obstacles.size() * 4;
	QuadVertex* vertexBuffer = new QuadVertex[Vertexes];
	for (uint32_t i = 0, j = 0; i < Vertexes; i+=4, j++)
	{
		vertexBuffer[i + 0].Position = obstacles[j].Min;
		vertexBuffer[i + 1].Position = glm::vec2(obstacles[j].Max.x, obstacles[j].Min.y);
		vertexBuffer[i + 2].Position = obstacles[j].Max;
		vertexBuffer[i + 3].Position = glm::vec2(obstacles[j].Min.x, obstacles[j].Max.y);
	}
	s_Data.BackgroundBuffer = CreateRef<VertexBuffer>(vertexBuffer, Vertexes * sizeof(QuadVertex));
	delete[] vertexBuffer;

	VertexBufferLayout Layout;
	Layout.Pushf(2); //Position
	Layout.Pushf(2); //Texture
	Layout.Pushf(3); //Tint Color
	s_Data.BackgroundVertexArray->AddBuffer(*s_Data.BackgroundBuffer, Layout);
}

void Renderer::SetParticleSize(float particleSize)
{
	s_Data.ParticleSize = particleSize;
	s_Data.QuadVertexPositions[0] = { -s_Data.ParticleSize, -s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[1] = {  s_Data.ParticleSize, -s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[2] = {  s_Data.ParticleSize,  s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[3] = { -s_Data.ParticleSize,  s_Data.ParticleSize, 0.0f, 1.0f };
}

void Renderer::BeginScene(const Camera& camera)
{
	s_CameraData.ViewProjectionMatrix = camera.GetViewProjectionMatrix();
	s_CameraData.CameraPosition = camera.GetPosition();
	s_CameraData.BBoxMax = glm::vec2(camera.GetPosition()) + camera.GetExtends() + glm::vec2(s_Data.ParticleSize);
	s_CameraData.BBoxMin = glm::vec2(camera.GetPosition()) - camera.GetExtends() - glm::vec2(s_Data.ParticleSize);
	s_CameraData.CameraWidth = camera.GetExtends().x * 2;

	StartBatch();
}

void Renderer::DrawQuad(const glm::vec2& position)
{
	DrawQuad(position, glm::vec3(1.0f));
}

void Renderer::DrawQuad(const glm::vec2& position, const glm::vec3& color)
{
	if (position.x < s_CameraData.BBoxMin.x || position.y < s_CameraData.BBoxMin.y || position.x > s_CameraData.BBoxMax.x || position.y > s_CameraData.BBoxMax.y)
	{
		return;
	}

	glm::mat4 transform = glm::translate(glm::mat4(1), { position.x, position.y, 1.0 });

	constexpr size_t quadVertexCount = 4;
	constexpr glm::vec2 textureCoords[] = { { 0.0f, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f } };

	if (s_Data.QuadIndexCount >= RendererData::MaxIndices)
		NextBatch();

	for (size_t i = 0; i < quadVertexCount; i++)
	{
		s_Data.QuadVertexBufferPtr->Position = transform * s_Data.QuadVertexPositions[i];
		s_Data.QuadVertexBufferPtr->UV = textureCoords[i];
		s_Data.QuadVertexBufferPtr->Color = color;
		s_Data.QuadVertexBufferPtr++;
	}

	s_Data.QuadIndexCount += 6;

	s_Stats.QuadCount++;
}

void Renderer::EndScene()
{
	Flush();
	DrawBackground();
}

void Renderer::ResetStats()
{
	memset(&s_Stats, 0, sizeof(Statistics));
}

Renderer::Statistics Renderer::GetStats()
{
	return s_Stats;
}

void Renderer::StartBatch()
{
	s_Data.QuadIndexCount = 0;
	s_Data.QuadVertexBufferPtr = s_Data.QuadVertexBufferBase;
}

void Renderer::NextBatch()
{
	Flush();
	StartBatch();
}

void Renderer::Flush()
{
	uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.QuadVertexBufferPtr - (uint8_t*)s_Data.QuadVertexBufferBase);
	s_Data.QuadVertexBuffer->SetData(s_Data.QuadVertexBufferBase, dataSize);

	s_Data.QuadShader->Bind();
	s_Data.QuadShader->SetUniformMat4f("MVPMatrix", s_CameraData.ViewProjectionMatrix);
	s_Data.QuadShader->SetUniform1f("Width", s_CameraData.CameraWidth);
	s_Data.QuadVertexArray->Bind();
	s_Data.QuadIndexBuffer->Bind();
	glDrawElements(GL_TRIANGLES, s_Data.QuadIndexCount, GL_UNSIGNED_INT, nullptr);

	s_Stats.DrawCalls++;
}

void Renderer::DrawBackground()
{
	s_Data.BackgroundShader->Bind();
	s_Data.BackgroundShader->SetUniformMat4f("MVPMatrix", s_CameraData.ViewProjectionMatrix);
	s_Data.BackgroundVertexArray->Bind();
	s_Data.BackgroundIndexBuffer->Bind();
	glDrawElements(GL_TRIANGLES, s_Data.BackgroundIndexBuffer->GetCount(), GL_UNSIGNED_INT, nullptr);
	s_Stats.DrawCalls++;
}
