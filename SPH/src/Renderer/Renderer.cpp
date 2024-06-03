
#include "Renderer.h"
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

struct ParticleVertex
{
	glm::vec2 Position;
	glm::vec2 UV;
	glm::vec3 Color;
};

struct QuadVertex
{
	glm::vec2 Position;
	glm::vec2 UV;
	glm::vec3 Color;
};

struct RendererData
{
	//Particle Data
	static const uint32_t MaxParticles = 20000;
	static const uint32_t MaxParticleVertices = MaxParticles * 4;
	static const uint32_t MaxParticleIndices = MaxParticles * 6;
	float ParticleSize = 5.f;

	Ref<VertexArray> ParticleVertexArray;
	Ref<VertexBuffer> ParticleVertexBuffer;
	Ref<IndexBuffer> ParticleIndexBuffer;
	Ref<Shader> ParticleShader;

	uint32_t ParticleIndexCount = 0;
	ParticleVertex* ParticleVertexBufferBase = nullptr; // store adres of cpu array
	ParticleVertex* ParticleVertexBufferPtr = nullptr; // store aders of next Particle in cpu array

	glm::vec4 ParticleVertexPositions[4];

	//Obstacles Data
	static const uint32_t MaxQuads = 1000;
	static const uint32_t MaxQuadVertices = MaxQuads * 4;
	static const uint32_t MaxQuadIndices = MaxQuads * 6;

	Ref<VertexArray> QuadVertexArray;
	Ref<VertexBuffer> QuadVertexBuffer;
	Ref<IndexBuffer> QuadIndexBuffer;
	Ref<Shader> QuadShader;

	uint32_t QuadCount = 0;
	uint32_t BackgroundQuadCount = 0;
	QuadVertex* QuadVertexBufferBase = nullptr;


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
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	//Init Data

	//Partice
	s_Data.ParticleVertexArray = CreateRef<VertexArray>();
	s_Data.ParticleVertexBuffer = CreateRef<VertexBuffer>(nullptr, s_Data.MaxParticleVertices * sizeof(ParticleVertex));
	VertexBufferLayout ParticelLayout;
	ParticelLayout.Pushf(2); //Position
	ParticelLayout.Pushf(2); //Texture
	ParticelLayout.Pushf(3); //Tint Color
	s_Data.ParticleVertexArray->AddBuffer(*s_Data.ParticleVertexBuffer, ParticelLayout);

	s_Data.ParticleVertexBufferBase = new ParticleVertex[s_Data.MaxParticleVertices];

	//precalculate Inices
	uint32_t* ParticleIndices = new uint32_t[s_Data.MaxParticleIndices];
	uint32_t offset = 0;
	for (uint32_t i = 0; i < s_Data.MaxParticleIndices; i += 6)
	{
		ParticleIndices[i + 0] = offset + 0;
		ParticleIndices[i + 1] = offset + 1;
		ParticleIndices[i + 2] = offset + 2;

		ParticleIndices[i + 3] = offset + 2;
		ParticleIndices[i + 4] = offset + 3;
		ParticleIndices[i + 5] = offset + 0;

		offset += 4;
	}
	s_Data.ParticleIndexBuffer = CreateRef<IndexBuffer>(ParticleIndices, s_Data.MaxParticleIndices);
	

	s_Data.ParticleVertexPositions[0] = { -s_Data.ParticleSize, -s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.ParticleVertexPositions[1] = {  s_Data.ParticleSize, -s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.ParticleVertexPositions[2] = {  s_Data.ParticleSize,  s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.ParticleVertexPositions[3] = { -s_Data.ParticleSize,  s_Data.ParticleSize, 0.0f, 1.0f };

	
	const std::string vertexSrc = 
	#include "Shaders/vertex.glsl"
	;

	const std::string fragmentSrc =
	#include "Shaders/fragment.glsl"
	;

	s_Data.ParticleShader = CreateRef<Shader>(vertexSrc, fragmentSrc);

	//Quad Data
	s_Data.QuadVertexArray = CreateRef<VertexArray>();
	s_Data.QuadVertexBuffer = CreateRef<VertexBuffer>(nullptr, s_Data.MaxQuadVertices * sizeof(QuadVertex));
	VertexBufferLayout QuadLayout;
	QuadLayout.Pushf(2);
	QuadLayout.Pushf(2);
	QuadLayout.Pushf(3);
	s_Data.QuadVertexArray->AddBuffer(*s_Data.QuadVertexBuffer, QuadLayout);

	s_Data.QuadVertexBufferBase = new QuadVertex[s_Data.MaxQuadVertices];

	s_Data.QuadIndexBuffer = CreateRef<IndexBuffer>(ParticleIndices, s_Data.MaxQuadIndices);

	const std::string fragmentBGSrc =
	#include "Shaders/fragmentBG.glsl"
	;

	s_Data.QuadShader = CreateRef<Shader>(vertexSrc, fragmentBGSrc);
	delete[] ParticleIndices;

}

void Renderer::Shutdown()
{
	delete[] s_Data.ParticleVertexBufferBase;
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

void Renderer::ResetRectangles()
{
	s_Data.QuadCount = 0;
	s_Data.BackgroundQuadCount = 0;
}

void Renderer::AddRectangle(const glm::vec2& min, const glm::vec2& max, const glm::vec3& color, bool bIsBackground)
{
	if (s_Data.QuadCount >= s_Data.MaxQuads)
	{
		return;
	}
	/*
	p4--p3
	|	|
	|	|
	p1--p2
	*/

	QuadVertex p1;
	p1.Position = glm::vec2(min);
	p1.Color = color;
	QuadVertex p2;
	p2.Position = glm::vec2(max.x, min.y);
	p2.Color = color;
	QuadVertex p3;
	p3.Position = glm::vec2(max);
	p3.Color = color;
	QuadVertex p4;
	p4.Position = glm::vec2(min.x, max.y);
	p4.Color = color;

	uint32_t index = s_Data.QuadCount * 4;

	s_Data.QuadVertexBufferBase[index + 0] = p1;
	s_Data.QuadVertexBufferBase[index + 1] = p2;
	s_Data.QuadVertexBufferBase[index + 2] = p3;
	s_Data.QuadVertexBufferBase[index + 3] = p4;

	if (bIsBackground)
	{
		s_Data.BackgroundQuadCount++;
	}
	s_Data.QuadCount ++;
}

void Renderer::UpdateRectangles()
{
	s_Data.QuadVertexBuffer->SetData(s_Data.QuadVertexBufferBase, (s_Data.QuadCount * 4) * sizeof(QuadVertex));
}

void Renderer::SetParticleSize(float particleSize)
{
	s_Data.ParticleSize = particleSize;
	s_Data.ParticleVertexPositions[0] = { -s_Data.ParticleSize, -s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.ParticleVertexPositions[1] = {  s_Data.ParticleSize, -s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.ParticleVertexPositions[2] = {  s_Data.ParticleSize,  s_Data.ParticleSize, 0.0f, 1.0f };
	s_Data.ParticleVertexPositions[3] = { -s_Data.ParticleSize,  s_Data.ParticleSize, 0.0f, 1.0f };
}

void Renderer::BeginScene(const Camera& camera)
{
	s_CameraData.ViewProjectionMatrix = camera.GetViewProjectionMatrix();
	s_CameraData.CameraPosition = camera.GetPosition();
	s_CameraData.BBoxMax = glm::vec2(camera.GetPosition()) + camera.GetExtends() + glm::vec2(s_Data.ParticleSize);
	s_CameraData.BBoxMin = glm::vec2(camera.GetPosition()) - camera.GetExtends() - glm::vec2(s_Data.ParticleSize);
	s_CameraData.CameraWidth = camera.GetExtends().x * 2;

	StartBatch();
	DrawBackground();
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

	if (s_Data.ParticleIndexCount >= RendererData::MaxParticleIndices)
		NextBatch();

	for (size_t i = 0; i < quadVertexCount; i++)
	{
		s_Data.ParticleVertexBufferPtr->Position = transform * s_Data.ParticleVertexPositions[i];
		s_Data.ParticleVertexBufferPtr->UV = textureCoords[i];
		s_Data.ParticleVertexBufferPtr->Color = color;
		s_Data.ParticleVertexBufferPtr++;
	}

	s_Data.ParticleIndexCount += 6;

	s_Stats.ParticleCount++;
}

void Renderer::EndScene()
{
	Flush();
	DrawObstacles();
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
	s_Data.ParticleIndexCount = 0;
	s_Data.ParticleVertexBufferPtr = s_Data.ParticleVertexBufferBase;

}

void Renderer::NextBatch()
{
	Flush();
	StartBatch();
}

void Renderer::Flush()
{
	uint32_t dataSize = (uint32_t)((uint8_t*)s_Data.ParticleVertexBufferPtr - (uint8_t*)s_Data.ParticleVertexBufferBase);
	s_Data.ParticleVertexBuffer->SetData(s_Data.ParticleVertexBufferBase, dataSize);

	s_Data.ParticleShader->Bind();
	s_Data.ParticleShader->SetUniformMat4f("MVPMatrix", s_CameraData.ViewProjectionMatrix);
	s_Data.ParticleShader->SetUniform1f("Width", s_CameraData.CameraWidth);
	s_Data.ParticleVertexArray->Bind();
	s_Data.ParticleIndexBuffer->Bind();
	glDrawElements(GL_TRIANGLES, s_Data.ParticleIndexCount, GL_UNSIGNED_INT, nullptr);

	s_Stats.DrawCalls++;
}

void Renderer::DrawBackground()
{
	//glEnable(GL_DEPTH_TEST);

	s_Data.QuadShader->Bind();
	s_Data.QuadShader->SetUniformMat4f("MVPMatrix", s_CameraData.ViewProjectionMatrix);
	s_Data.QuadVertexArray->Bind();
	s_Data.QuadIndexBuffer->Bind();
	glDrawElementsBaseVertex(GL_TRIANGLES, s_Data.BackgroundQuadCount * 6, GL_UNSIGNED_INT, nullptr, 0);
	s_Stats.DrawCalls++;

	//glDisable(GL_DEPTH_TEST);
}

void Renderer::DrawObstacles()
{
	s_Data.QuadShader->Bind();
	s_Data.QuadShader->SetUniformMat4f("MVPMatrix", s_CameraData.ViewProjectionMatrix);
	s_Data.QuadVertexArray->Bind();
	s_Data.QuadIndexBuffer->Bind();
	glDrawElementsBaseVertex(GL_TRIANGLES, (s_Data.QuadCount - s_Data.BackgroundQuadCount) * 6, GL_UNSIGNED_INT, nullptr, s_Data.BackgroundQuadCount * 4);
	s_Stats.DrawCalls++;
}
