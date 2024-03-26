
#include "Renderer.h"
#include "Texture.h"
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
};

struct RendererData
{
	static const uint32_t MaxQuads = 20000;
	static const uint32_t MaxVertices = MaxQuads * 4;
	static const uint32_t MaxIndices = MaxQuads * 6;

	Ref<VertexArray> QuadVertexArray;
	Ref<VertexBuffer> QuadVertexBuffer;
	Ref<IndexBuffer> QuadIndexBuffer;
	Ref<Shader> QuadShader;
	Ref<Texture> QuadTexture;

	uint32_t QuadIndexCount = 0;
	QuadVertex* QuadVertexBufferBase = nullptr; // store adres of cpu array
	QuadVertex* QuadVertexBufferPtr = nullptr; // store aders of next Quad in cpu array

	glm::vec4 QuadVertexPositions[4];
};

static RendererData s_Data;

glm::mat4 Renderer::s_ViewProjectionMatrix = glm::mat4(1);
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
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//Init Data
	s_Data.QuadVertexArray = CreateRef<VertexArray>();
	s_Data.QuadVertexBuffer = CreateRef<VertexBuffer>(nullptr, s_Data.MaxVertices * sizeof(QuadVertex));
	VertexBufferLayout Layout;
	Layout.Pushf(2); //Position
	Layout.Pushf(2); //Texture
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

	s_Data.QuadVertexPositions[0] = { -5.0f, -5.0f, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[1] = {  5.0f, -5.0f, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[2] = {  5.0f,  5.0f, 0.0f, 1.0f };
	s_Data.QuadVertexPositions[3] = { -5.0f,  5.0f, 0.0f, 1.0f };

	s_Data.QuadShader = CreateRef<Shader>("src\\Shaders\\vertex.glsl", "src\\Shaders\\fragment.glsl");
	s_Data.QuadTexture = CreateRef<Texture>("dot.png");
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

void Renderer::BeginScene(const Camera& camera)
{
	s_ViewProjectionMatrix = camera.GetViewProjectionMatrix();

	StartBatch();
}

void Renderer::DrawQuad(const glm::vec2& position)
{
	glm::mat4 transform = glm::translate(glm::mat4(1), { position.x, position.y, 1.0 });

	constexpr size_t quadVertexCount = 4;
	constexpr glm::vec2 textureCoords[] = { { 0.0f, 0.0f }, { 1.0f, 0.0f }, { 1.0f, 1.0f }, { 0.0f, 1.0f } };

	if (s_Data.QuadIndexCount >= RendererData::MaxIndices)
		NextBatch();

	for (size_t i = 0; i < quadVertexCount; i++)
	{
		s_Data.QuadVertexBufferPtr->Position = transform * s_Data.QuadVertexPositions[i];
		s_Data.QuadVertexBufferPtr->UV = textureCoords[i];
		s_Data.QuadVertexBufferPtr++;
	}

	s_Data.QuadIndexCount += 6;

	s_Stats.QuadCount++;
}

void Renderer::EndScene()
{
	Flush();
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
	s_Data.QuadShader->SetUniform1i("ColorMap", 1);
	s_Data.QuadShader->SetUniformMat4f("MVPMatrix", s_ViewProjectionMatrix);

	s_Data.QuadTexture->Bind(1);
	s_Data.QuadVertexArray->Bind();
	s_Data.QuadIndexBuffer->Bind();
	glDrawElements(GL_TRIANGLES, s_Data.QuadIndexCount, GL_UNSIGNED_INT, nullptr);

	s_Stats.DrawCalls++;
}
