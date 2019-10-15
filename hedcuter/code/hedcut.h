/*
Wikipedia:

Hedcut is a term referring to a style of drawing, associated with The Wall Street Journal half-column portrait illustrations.
They use the stipple method of many small dots and the hatching method of small lines to create an image, and are designed to
emulate the look of woodcuts from old-style newspapers, and engravings on certificates and currency.
The phonetic spelling of "hed" may be based on newspapers' use of the term hed for "headline."
*/

#pragma once

#include "wcvt.h" //weighted centroidal voronoi tessellation

struct HedcutDisk
{
	cv::Point2d center;
	float radius;
	cv::Scalar color;
};

class Hedcut
{
public:

	Hedcut();

	bool build(cv::Mat & input_image, int n);

	const std::list<HedcutDisk> & getDisks() const { return disks;  }

	//cvf control flags
	int cvt_iteration_limit; //max number of iterations when building cvf
	float max_site_displacement; //max tolerable site displacement in each iteration.
	bool average_termination;	//ture when the algorithm terminates with average displacement, not max displacement

	bool debug; //if true, debug information will be excuted

private:

	void sample_initial_points(cv::Mat & img, int n, std::vector<cv::Point2d> & pts);
	void create_disks(cv::Mat & img, CVT & cvt);

	std::list<HedcutDisk> disks;
};
