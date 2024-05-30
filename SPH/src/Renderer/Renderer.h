#pragma once

#include "Camera.h"
#include "Shader.h"
#include <vector>

struct Obstacle
{
	glm::vec2 Min = glm::vec2(0);
	glm::vec2 Max = glm::vec2(0);
};

class Renderer
{
public:
	static void Init();
	static void Shutdown();

	static void SetColor(float r, float g, float b, float a = 1.0f);
	static void Resize(int width, int height);
	static void Clear();
	static void SetObstacles(const std::vector<Obstacle>& obstacles);
	static void SetParticleSize(float particleSize);

	static void BeginScene(const Camera& camera);
	static void DrawQuad(const glm::vec2& position);
	static void DrawQuad(const glm::vec2& position, const glm::vec3& color);
	static void EndScene();

	struct Statistics
	{
		uint32_t DrawCalls = 0;
		uint32_t QuadCount = 0;

		uint32_t GetTotalVertexCount() const { return QuadCount * 4; }
		uint32_t GetTotalIndexCount() const { return QuadCount * 6; }
	};

	static void ResetStats();
	static Statistics GetStats();

private:
	static void StartBatch();
	static void NextBatch();
	static void Flush();
	static void DrawBackground();

};