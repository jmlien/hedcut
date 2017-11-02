#include "wcvt_gpu.h"

std::vector<VorCell> CVT::cells;

float rotateY;
float translateZ;
cv::Mat input_image, grayscale, root;
int iteration = 0;

void idle_GPU(void)
{
	glutPostRedisplay();
}
void keyboard_GPU(unsigned char key, int x, int y)
{
	switch (key)
	{
		case 'r':
			rotateY += 10.0;
			if (rotateY > 360 || rotateY < -360) rotateY = 0.0;
		break;
		translateZ += 1.0;
		if (translateZ > 1.0) translateZ = 0.0;
		default:
		break;
	}
}
float getDepthValue(int x, int y)
{
	if (x > input_image.size().width || y > input_image.size().height) std::cout << "Access violation with depth buffer" << std::endl;
	float depth = 0.0f;

	glReadBuffer(GL_FRONT);
	glReadPixels(x, y, 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &depth);
	return depth;
}

void CVT::compute_weighted_cvt_GPU(cv::Mat &  img, std::vector<cv::Point2d> & sites)
{
	//init 
	int site_size = sites.size();
	cells.resize(site_size);
	for (int i = 0; i < site_size; i++)
	{
		cells[i].site = sites[i];
	}

	float max_dist_moved = FLT_MAX;

	run_GPU(argc_GPU, argv_GPU, img);
}

void CVT::run_GPU(int argc, char**argv, cv::Mat& img)
{
	//Init opengl
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_DOUBLE | GLUT_DEPTH);
	glutInitWindowSize(img.size().width, img.size().height);
	glutInitWindowPosition(50, 50);
	glutCreateWindow("Image");

	init_GPU(img);

	glutDisplayFunc(display_GPU);
	glutKeyboardFunc(keyboard_GPU);
	glutIdleFunc(idle_GPU);

	glutMainLoop();
}


float CVT::move_sites_GPU()
{
	float max_offset = 0;

	float total = 0;
	cv::Point2d new_pos(0, 0);
	return max_offset;
}

//buil the VOR once
void CVT::vor_GPU()
{
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glClearDepth(1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(-1.0, 1.0, -1.0, 1.0, 1.0, -1.0);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glRotatef(rotateY, 0.0, 1.0, 0.0);

	//Cone information
	GLdouble base = 50.0 ;		//*****
	GLdouble height = 1.0;	//*****
	GLint slices = 50;		//*****
	GLint stacks = 50;

	//Image information
	GLfloat d = 0.0;
	unsigned int r = 0, g = 0, b = 0;
	GLfloat red = 0.0, green = 0.0, blue = 0.0;

	//Draw discrete voronoi diagram
	for (int i = 0; i < cells.size(); i++)
	{
		cv::Point pix(cells[i].site.x, cells[i].site.y);
		root.at<ushort>(pix.x, pix.y) = i;
		d = (256 - (float)grayscale.at<uchar>(pix.x, pix.y))*1.0f / 256;
		
		r = input_image.at<cv::Vec3b>(pix.x, pix.y)[2];
		g = input_image.at<cv::Vec3b>(pix.x, pix.y)[1];
		b = input_image.at<cv::Vec3b>(pix.x, pix.y)[0];
		
		red = (float)r / 255.0;
		green = (float)g / 255.0;
		blue = (float)b / 255.0;

		
		glPushMatrix();
		//Convert opengl coordinates to opencv coordinates
		glScalef(2.0 / (float)input_image.size().width, 2.0 / (float)input_image.size().height, 1.0);
		glTranslatef(-(float)input_image.size().width / 2.0, (float)input_image.size().height / 2.0, 0.0);
		glRotatef(180.0, 1.0, 0.0, 0.0);

		glTranslatef(cells[i].site.y*1.0f, cells[i].site.x*1.0f, -d);
		glColor3f(red, green, blue);
		glutSolidCone(base, height, slices, stacks);
		glPopMatrix();
	}

}


void CVT::init_GPU(cv::Mat& img)
{
	glEnable(GL_DEPTH_TEST);

	//Initialize global variables
	rotateY = 0.0;
	translateZ = 0.0;
	input_image = img;
	cv::cvtColor(input_image, grayscale, CV_BGR2GRAY);
	root = cv::Mat(img.size(), CV_16U, cv::Scalar::all(USHRT_MAX)).clone();

	
}
void CVT::display_GPU(void)
{
	vor_GPU();
	move_sites_GPU();

	iteration++;
	glutSwapBuffers();
}

