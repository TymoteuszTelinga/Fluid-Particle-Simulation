#pragma once

#include <glm/glm.hpp>

class Camera
{
public:
	Camera(float width, float height);

	void Resize(float width, float height);
	void Zoom(float delta);

	const glm::vec3& GetPosition() const { return m_Position; }
	void SetPosition(const glm::vec3& position) { m_Position = position; RecalculateViewMatrix(); }

	float GetRotation() const { return m_Rotation; }
	void SetRotation(float rotation) { m_Rotation = rotation; RecalculateViewMatrix(); }

	const glm::mat4& GetProjectionMatrix() const { return m_ProjectionMatrix; }
	const glm::mat4& GetViewMatrix() const { return m_ViewMatrix; }
	const glm::mat4& GetViewProjectionMatrix() const { return m_ViewProjectionMatrix; }

	float GetZoomLevel() const { return m_ZoomLevel; };
	glm::vec2 GetExtends() const {return glm::vec2(m_AspectRatio*m_HalfHeight * m_ZoomLevel, m_HalfHeight * m_ZoomLevel); };

private:
	void RecalculateViewMatrix();
	void RecalculateProjectionMatrix();

private:
	glm::mat4 m_ProjectionMatrix;
	glm::mat4 m_ViewMatrix;
	glm::mat4 m_ViewProjectionMatrix;

	glm::vec3 m_Position = { 0.0f, 0.0f, 0.0f };
	float m_Rotation = 0.0f;

	float m_AspectRatio = 0.0f;
	float m_HalfHeight;
	float m_ZoomLevel = 1.f;
};

