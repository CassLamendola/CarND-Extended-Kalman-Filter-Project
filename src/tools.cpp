#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  	VectorXd rmse(4);
  	rmse << 0,0,0,0;
  
  	if(estimations.size()==0 || estimations.size()!=ground_truth.size())
	{
	    return rmse;
	}
	//accumulate squared residuals
	for(int i=0; i < estimations.size(); ++i){
        VectorXd residual = estimations[i] - ground_truth[i];
        residual = residual.array() * residual.array();
		rmse += residual;
	}
	rmse = rmse/estimations.size();
	rmse = rmse.array().sqrt();
	return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {
  	MatrixXd Hj(3,4);
	
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//check for division by zero
	if (px == 0 || py == 0)
	{
		cout << "CalculateJacobian() Error - division by 0";
	    return Hj;
	}

	//compute the Jacobian matrix
	float c1 = pow(px, 2)+pow(py,2);
	float c2 = sqrt(c1);
	float c3 = pow(c1, (3/2));
	
	Hj << (px/c2), (py/c2), 0, 0,
        (-1*py/c1), (px/c1), 0, 0,
        py*((vx*py) - (vy*px))/c3, px*((vy*px) - (vx*py))/c3, px/c2, py/c2;

    return Hj;
}
