#include "Camera.h"
#include <glm/gtc/matrix_transform.hpp>

Camera::Camera(float width, float height)
	: m_ViewMatrix(1.0f)
{
	m_AspectRatio = width / height;
	m_HalfHeight = height / 2.f;

	RecalculateProjectionMatrix();
}

void Camera::Resize(float width, float height)
{
	m_AspectRatio = width / height;
	m_HalfHeight = height / 2.f;
	RecalculateProjectionMatrix();
}

void Camera::Zoom(float delta)
{
	m_ZoomLevel += delta;
	m_ZoomLevel = std::max(m_ZoomLevel, 0.1f);
	RecalculateProjectionMatrix();
}

void Camera::RecalculateViewMatrix()
{
	glm::mat4 transform = glm::translate(glm::mat4(1.0f), m_Position) * glm::rotate(glm::mat4(1.0f), glm::radians(m_Rotation), glm::vec3(0, 0, 1));

	m_ViewMatrix = glm::inverse(transform);
	m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
}

void Camera::RecalculateProjectionMatrix()
{
	m_ProjectionMatrix = glm::ortho(-m_AspectRatio * m_HalfHeight * m_ZoomLevel, m_AspectRatio * m_HalfHeight *m_ZoomLevel, -m_HalfHeight * m_ZoomLevel, m_HalfHeight * m_ZoomLevel, -1.f, 1.f);
	m_ViewProjectionMatrix = m_ProjectionMatrix * m_ViewMatrix;
}
