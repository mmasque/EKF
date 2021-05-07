# EKF

An EKF implementation for the Monash Nova Rover team. Yields position estimate by fusing control input data, camera data, and AR tag locations. 
Designed for use in the Australian Rover Challenge competition.

Based on ekf code by: Atsushi Sakai (@Atsushi_twi)
Originally adapted for NovaRover by Jack McRobbie, Cheston Chow and Peter Shi

This version modifies the EFK to work without a GPS, fixes multiple issues, and adds functionality to handle AR tag location inputs. 
