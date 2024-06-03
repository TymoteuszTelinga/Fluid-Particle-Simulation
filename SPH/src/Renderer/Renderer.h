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

	/*
	* set clear color of frame buffer
	* @param r red chanel of new color
	* @param g green chanel of new color
	* @param b blue chanel of new color
	* @param a alpha chanel of new color
	*/
	static void SetColor(float r, float g, float b, float a = 1.0f);
	/*
	* changes the resolution of the rendered image
	* @param width width of new image
	* @param height height of new image
	*/
	static void Resize(int width, int height);
	/*
	* clear the fram buffer
	*/
	static void Clear();
	static void SetParticleSize(float particleSize);

	/*
	* clear list of static objects
	*/
	static void ResetRectangles();
	/*
	* Add rectangle to the list of static objects
	* @param min position of the lower left corner of the rectangle
	* @param max position of the upper right corner of the rectangle
	* @param color color of the rectangle
	* @param bIsBackgroun specifies whether to draw a given rectangle as a background
	*/
	static void AddRectangle(const glm::vec2& min, const glm::vec2& max, const glm::vec3& color = glm::vec3(1.f), bool bIsBackground = false);
	/*
	* update GPU buffer of static objects
	*/
	static void UpdateRectangles();

	/*
	* Starts drawing of a new frame
	* @param camera camera with which the user observes the world
	*/
	static void BeginScene(const Camera& camera);
	/*
	* adds a particle to the batch for rendering
	* @param position particle position in the world
	*/
	static void DrawQuad(const glm::vec2& position);
	/*
	* adds a particle to the batch for rendering
	* @param position particle position in the world
	* @param color particle color
	*/
	static void DrawQuad(const glm::vec2& position, const glm::vec3& color);
	/*
	* represents the end of drawing of a given frame
	*/
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
	static void DrawObstacles();

};