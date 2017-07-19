/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>
#include <limits>


#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.

	// Define the number of particles
	num_particles = 100;

	// Retrieve standard deviations
	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// Create gaussian distributions for each parameter
	default_random_engine gen;
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);
	
	// Create particles based on initial measurements and uncertainty
	for (int i=0; i<num_particles; ++i)
	{
		Particle p;
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
	}

	// Initialization is complete
	is_initialized = true;

}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.

	// Create gaussian distributions for each parameter
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);
	
	for (int i=0; i<num_particles; ++i)
	{
		if (abs(yaw_rate) > 0.00001)
		{
			particles[i].x += velocity / yaw_rate * (sin(particles[i].theta + yaw_rate * delta_t) - sin(particles[i].theta)) + dist_x(gen);
			particles[i].y += velocity / yaw_rate * (cos(particles[i].theta) - cos(particles[i].theta + yaw_rate * delta_t)) + dist_y(gen);
		}
		else
		{
			particles[i].x += velocity * delta_t * cos(particles[i].theta) + dist_x(gen);
			particles[i].y += velocity * delta_t * sin(particles[i].theta) + dist_y(gen);
		}
		particles[i].theta += delta_t * yaw_rate + dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.

	// Loop over all observations to check closest predicted landmark
	for (int i=0; i<observations.size(); ++i)
	{
		double obs_x = observations[i].x;
		double obs_y = observations[i].y;

		double closest_distance = std::numeric_limits<double>::max();
		double current_distance = 0.;

		// Find closest predicted landmark
		for (int j=0; j<predicted.size(); ++i)
		{
			current_distance = dist(obs_x, obs_y, predicted[j].x, predicted[j].y);
			if (current_distance < closest_distance)
			{
				closest_distance = current_distance;
				observations[i].id = predicted[j].id; // save landmark id
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

	double std_x = std_landmark[0];
	double std_y = std_landmark[1];
	
	// Loop over each particle
	for (int i=0; i<num_particles; ++i)
	{
		Particle particle = particles[i];
		particle.weight = 1.;  // initial value

		for (int j=0; j<observations.size(); ++j)
		{
			// Extract observation data
			double obs_x = observations[j].x;
			double obs_y = observations[j].y;

			// Convert observation to map coordinates
			double obs_map_x = obs_x * cos(particle.theta) - obs_y * sin(particle.theta) + particle.x;
			double obs_map_y = obs_x * sin(particle.theta) + obs_y * cos(particle.theta) + particle.y;

			// Identify closest landmark to observation
			double closest_distance = 1.e50;
			double current_distance = 0.;
			double delta_x = 0.;
			double delta_y = 0.;
			for (int k=0; k<map_landmarks.landmark_list.size(); ++k)
			{
				double landmark_x = map_landmarks.landmark_list[k].x_f;
				double landmark_y = map_landmarks.landmark_list[k].y_f;
				current_distance = dist(obs_map_x, obs_map_y, landmark_x, landmark_y);
				if (current_distance < closest_distance)
				{
					// Update closest_distance and delta coordinates (for calculating weight later)
					closest_distance = current_distance;
					delta_x = obs_map_x - landmark_x;
					delta_y = obs_map_y - landmark_y;
				}
			}

			// Update weight with current observation (no need to multiply by a constant since we will normalize)
			particle.weight *= exp(- (delta_x * delta_x) / (2 * std_x * std_x) - (delta_y * delta_y) / (2 * std_y * std_y));

		}

		particles[i] = particle;
	}

	// Calculate sum of all weights
	double sum_weight = 0.;
	for (int i=0; i<num_particles; ++i)
	{
		sum_weight += particles[i].weight;
	}

	// Normalize weights
	for (int i=0; i<num_particles; ++i)
	{
		particles[i].weight /= sum_weight;
	}
};


void ParticleFilter::resample() {
	// Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	// Extract all weights
	std::vector<double> weight_array(num_particles);
	for (int i=0; i<num_particles; ++i)
	{
		weight_array[i] = particles[i].weight;
	}	
	
	// Create distribution
	default_random_engine gen;
	discrete_distribution<int> dist (weight_array.begin(), weight_array.end());

	// Sample from distribution
	std::vector<Particle> new_particles;

	for (int i=0; i<num_particles; ++i)
	{
		// Select particle from distribution
		int next_index = dist(gen);
		Particle new_particle = particles[next_index];
		new_particles.push_back(new_particle);
	}
	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
