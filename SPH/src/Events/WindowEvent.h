#pragma once
#include "Event.h"
#include <sstream>

class WindowResizeEvent : public Event
{
public:
	WindowResizeEvent(unsigned int width, unsigned int height)
		: m_Width(width), m_Height(height) {}

	unsigned int GetWidth() const { return m_Width; }
	unsigned int GetHeight() const { return m_Height; }

	std::string ToString() const override
	{
		std::stringstream ss;
		ss << "WindowResizeEvent: " << m_Width << ", " << m_Height;
		return ss.str();
	}

	EVENT_CLASS_TYPE(WindowResize)

private:
	unsigned int m_Width, m_Height;
};

class ScrollEvent : public Event
{
public:
	ScrollEvent(double xoff, double yoff)
		: m_Xoffset(xoff), m_Yoffset(yoff) {}

	double GetX() const { return m_Xoffset; }
	double GetY() const { return m_Yoffset; }

	std::string ToString() const override
	{
		std::stringstream ss;
		ss << "ScrollEvent: " << m_Xoffset << ", " << m_Yoffset;
		return ss.str();
	}

	EVENT_CLASS_TYPE(Scroll)

private:
	double m_Xoffset, m_Yoffset;
};