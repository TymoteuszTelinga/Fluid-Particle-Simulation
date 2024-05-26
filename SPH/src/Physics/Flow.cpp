#include "Flow.h"


void Flow::in(size_t amount, Ref<Particles> particles) const{
	for (int index = 0; index < amount; index++) {
		if (particles->getSize() >= particles->getCapacity()) {
			return;
		}
		float x_pos = random(in_area.x_pos, in_area.x_pos + in_area.width);
		float y_pos = random(in_area.y_pos, in_area.y_pos + in_area.heigth);

		particles->addParticle(x_pos, y_pos);
	}
}

void Flow::out(Ref<Particles> particles) const {
	for (int index = particles->getSize(); index >= 0; index--) {
		glm::vec2 pos = particles->getPosition(index);
		if (pos.x > out_area.x_pos && pos.x < out_area.x_pos + out_area.width &&
			pos.y > out_area.y_pos && pos.y < out_area.y_pos + out_area.heigth) {
			particles->remove(index);
		}
	}
}

float Flow::random(float from, float to) const {
	return from + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (to - from)));
}