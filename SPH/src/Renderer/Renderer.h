#pragma once

#include "Camera.h"
#include "Shader.h"
#include "Core/CommonTypes.h"
#include <vector>


class Renderer
{
public:
	static void Init();
	static void Shutdown();

	static void SetColor(float r, float g, float b, float a = 1.0f);
	static void Resize(int width, int height);
	static void Clear();
	static void SetParticleSize(float particleSize);

	static void ResetRectangles();
	static void AddRectangle(const glm::vec2& min, const glm::vec2& max, const glm::vec3& color = glm::vec3(1.f));
	static void UpdateRectangles();

	static void BeginScene(const Camera& camera);
	static void DrawQuad(const glm::vec2& position);
	static void DrawQuad(const glm::vec2& position, const glm::vec3& color);
	static void EndScene();

	struct Statistics
	{
		uint32_t DrawCalls = 0;
		uint32_t ParticleCount = 0;

		uint32_t GetTotalVertexCount() const { return ParticleCount * 4; }
		uint32_t GetTotalIndexCount() const { return ParticleCount * 6; }
	};

	static void ResetStats();
	static Statistics GetStats();

private:
	static void StartBatch();
	static void NextBatch();
	static void Flush();
	static void DrawBackground();

};