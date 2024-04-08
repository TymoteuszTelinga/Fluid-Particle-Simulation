#include "Density.h"
#include <time.h>
#include <stdio.h>

void Density::Calculate(std::vector<Particle>& particles) {
	for (int i = 0; i < particles.size(); i++) {
		Particle& center = particles[i];
		center.AddPartialDensity(Particle::MASS * KERNEL(0.0f, KERNEL_RADIUS));
		for (int j = i+1; j < particles.size(); j++) {
			Particle& otherParticle = particles[j];
			float distance = center.calculateDistance(otherParticle);
			float density = Particle::MASS*KERNEL(distance, KERNEL_RADIUS);
			center.AddPartialDensity(density);
			otherParticle.AddPartialDensity(density);
		}
	}
}


