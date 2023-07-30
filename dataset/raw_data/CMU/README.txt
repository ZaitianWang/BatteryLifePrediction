-------------------
GENERAL INFORMATION
-------------------

1. Title of Dataset: eVTOL Battery Dataset

2. Author Information

Author Contact Information
    Name: Alexander Bills
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Ave, Pittsburgh, PA, 15213
    Email: abills@andrew.cmu.edu

Author Contact Information
    Name: Shashank Sripad
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Avenue, Pittsburgh, Pennsylvania, USA, 15213
    Email: ssripad@cmu.edu

Author Contact Information
	Name: Leif Fredericks
	Email: leif.fredericks@gmail.com

Author Contact Information
	Name: Matthew Guttenberg
	Institution: Carnegie Mellon University
	Address: 5000 Forbes Ave, Pittsburgh, PA, 15213
	Email: mguttenb@andrew.cmu.edu

Author Contact Information
	Name: Devin Charles
	Email: devinccharles@gmail.com

Author Contact Information
	Name: Evan Frank
	Email: evanericfrank@gmail.com

Author Contact Information
    Name: Venkat Viswanathan
    Institution: Carnegie Mellon University
    Address: 5000 Forbes Ave, Pittsburgh, PA, 15213
    Email: venkvis@cmu.edu

---------------------
DATA & FILE OVERVIEW
---------------------

Each file contains the data from experimental testing from one cell throughout that cell’s life. The experimental protocols, further explanation of which can be found in the above paper, are summarized below:

VAH01.csv: Baseline
VAH02.csv: Extended cruise (1000 sec)
VAH05.csv: 10% power reduction during discharge (flight)
VAH06.csv: CC charge current reduced to C/2
VAH07.csv: CV charge voltage reduced to 4.0V
VAH09.csv: Thermal chamber temperature of 20 degrees C
VAH10.csv: Thermal chamber temperature of 30 degrees C
VAH11.csv: 20% power reduction during discharge (flight)
VAH12.csv: Short cruise length (400 sec)
VAH13.csv: Short cruise length (600 sec)
VAH15.csv: Extended cruise (1000 sec)
VAH16.csv: CC charge current reduced to 1.5C
VAH17.csv: Baseline
VAH20.csv: Charge current reduced to 1.5C
VAH22.csv: Extended cruise (1000 sec)
VAH23.csv: CV charge voltage reduced to 4.1V
VAH24.csv: CC charge current reduced to C/2
VAH25.csv: Thermal chamber temperature of 20 degrees C
VAH26.csv: Short cruise length (600 sec)
VAH27.csv: Baseline
VAH28.csv: 10% power reduction during discharge (flight)
VAH30.csv: Thermal chamber temperature of 35 degrees C

There are a few issues in the dataset here, and some are described here (this list may not be exhaustive, but should account for most issues):

VAH05
	- Missing a Capacity Test after cycle 1000, and the rest period from the preceding cycle.
VAH09
	- Apparent Tester Malfunction at Cycle 64 (cell did not charge all the way to 4.2V)
	- Apparent Tester Malfunction at Cycle 92 -- appears to be a discontinuity during the cycle
	- Cycle 154 has an issue - looks like a capacity test was started, then stopped early
	- Tester Malfunction at Cycle 691 - same as what happened at cycle 92 --appears to be a discontinuity during the cycle
VAH10
	- Apparent Malfunction at cycle 248: data stops during mission discharge cycle
	- Apparent Malfunction at cycle 631: data stops during mission discharge cycle, followed by a long break
	- During Cycle 735, Data collection stops, then resumes after a period of rest
	- During Cycle 1151, Data collection stops, then resumes after a period of rest
VAH11
	- Apparent Malfunction at cycle 817 data stops during capacity test
	- Some missing data during cycle 1898
VAH13
	- Apparent Malfunction at cycle 816/817 - missing capacity test
VAH25
	- Ran 2 Cap Tests in a row at Cycle 461/462
VAH26
	- Apparent Malfunction at cycle 872/873 - Looks like data stopped and started during the rest period following a mission cycle
VAH27
	- Unphysical voltage Spike During Cycle 20
	- Cycle Cut during 256/257 - CV charge is in a separate File
	- Very long rest time in cycle 585
VAH28
	- Ran 2 Cap Tests in a row at Cycle 256/257
	- Apparent Malfunction at cycle 619/620 - Looks like data stopped and started during the CV charging segment
	- Same as above malfunction in 1066/1067

Also note that the cycle number is not accurate. The cycle number as recorded is the raw number from the cell testers, which output multiple files, while in this dataset we have combined those files for ease of manipulation.

-----------------------------------------
DATA DESCRIPTION FOR ALL FILES
-----------------------------------------

1. Number of variables: 10

2. Number of cases/rows: Varies

4. Variable List

    A. Name: time_s
       Description: Time since beginning of experiment in seconds

    B. Name: Ecell_V
       Description: Cell Voltage

	C. Name: I_mA
       Description: Cell current in milliamperes

	D. Name: EnergyCharge_W_h
	   Description: The amount of energy supplied to the cell during charge in watt-hours

	E. Name: QCharge_mA_h
	   Description: The amout of charge supplied to the cell during charge in milliampere-hours

	F. Name: EnergyDischarge_W_h
	   Description: The amount of energy extracted from the cell during discharge in watt-hours

	G. Name: QDischarge_mA_h
   	   Description: The amount of charge extracted from the cell during discharge in milliampere-hours

	H. Name: Temperature__C
	   Description: Cell surface temperature in degrees celcius

	I. Name: cycleNumber
	   Description: Cycle number as recorded by the cell tester (see above note)

	J. Name: Ns
	   Description: Cycle segment (varies, useful for discerning between segments)
--------------------------
METHODOLOGICAL INFORMATION
--------------------------

1. Software-specific information:
N/A

2. Equipment-specific information:

Manufacturer: Sony-Murata
Model: 18650 VTC-6 cell

Manufacturer: Arbin
Model: 200A cell holder

Manufacturer: BioLogic
Model: BCS-815 Modular Battery Cycler

3. Date of data collection (single date, range, approximate date): 2018-2019
