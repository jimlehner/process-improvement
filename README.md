# process_improvement.py
The `process_improvement.py` library (version 1.1.3) is a collection of modules and functions designed to help identify, understand, and eliminate the influence of the two types of variation (common causes of routine variation and assignable causes of exceptional variation) that influence business and manufacturing processes. The purpose of this library is to provide manufacturing, quality, and process engineers with a free alternative to subscription based software like JMP and Minitab. While both of these software packages provide users with analytical tools capable of making sense of variation, they also divorce users from a deeper understanding of the analysis of data produced by processes. 

The primary tool of the `process_improvement.py` package is the process behavior chart for individual values and a moving range called the XmR Chart. The `process_improvement.py` contains additional modules and functions related to the task of process improvement including capability analysis, network analysis, comparison charts, and limit charts. 

The `process_improvement.py` package is part of the larger body of work called `The Broken Quality Initiative` (BrokenQuality.com). The aim of BQI is to address industries' pervasive lack of knowledge regarding variation and the only tool capable of making sense of variation, the process behavior chart (control chart). 

Visit [BrokenQuality.com](https://www.BrokenQuality.com/bookshelf) for resources and more details regarding the application and use of `process behavior charts`. Contact me **James.Lehner@gmail.com** if you have questions or would like to collaborate. 

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation) 
- [Usage](#usage)
- [Modules](#modules)
- [Contributing](#contributing) 
- [Contact](#contact) 
- [License](#license)
- [Additional Information](#additional-information)

## Prerequisites
Before you begin, ensure you have met the following requirements: 
- You have installed [Python](https://www.python.org/) 3.6 or higher. 
- You have a working knowledge of Python and data analysis libraries such as pandas and matplotlib. 
- You have a working knowledge of `Process Behavior Charts` and `Statistical Process Control`. 

## Installation
To install `process_improvement.py` directly from GitHub, enter the following command using the `command prompt`:

```pip install git+https://github.com/jimlehner/process_improvement```

## Usage
After `installation` the `process_improvement.py` library can be used as follows:
1. From process_improvement import xmr_charts module as xmr:
```from process_improvement import xmr_charts as xmr```
2. Call the function of interest from the requisite module:
```xmr.xmrchart(df, 'Values', 'Observations', title='Example X-chart')```

## Modules
The `process_improvement.py` pacakge contains 5 modules:
1. xmr_charts
2. process_capability
3. comparison_charts
4. limit_charts
5. network_analysis
Each of these modules can be used to address a different aspect of understanding variation. 

## Functions
```xmr_charts.py```
The `xmr_charts` module contains 3 function:
1. `xmr_chart`: Generates a process behavior chart of individual values and a moving range called an XmR Chart from the provided DataFrame.
2. `xchart`: Generates the X Chart portion of an XmR Chart from the provided DataFrame.
3. `mrchart`: Generates the moving range (mR) Chart portion of an XmR Chart from the provided DataFrame.

```process_capability```
The `process_capability` module contains 3 functions:
1. `capability_histogram`: Generates a capability histogram of the provided process data in the context of the specifciation limits and the option of displaying the process capability indices.
2. `multi_chart`: Generates the X Chart portion of an XmR Chart and a capability histogram in the same figure to enable direct visual comparison of process behavior and the distribution of the data.
3. `process_capabilities`: Calculates the process capability indices of Cp, Cpk, Pp, and Ppk, based on the provided process data, upper and lower specification limits, and target value.

```comparison_charts```
The `comparison_charts` module contains 3 functions:
1. `xchart_comparison`: Generates X Charts from the provided list of DataFrames and visually compares their statistics.
2. `mrchart_comparison`: Generates moving range (mR) Charts from the provided list of DataFrames and visually compares their statistics.
3. `xmr_comparison`: Dynamically generates a grid of subplots composed of XmR Charts from the provided list of DataFrames. Comparison is limited to a list length of 5.

```limit_charts```
The `limit_chart` module contains 1 function:
1. `limit_chart`: Generates a limit chart that sequentially plots process data in the context of the provided specification limits and target.

```network_analysis```
The `network_analysis` module contains 2 function:
1. `network_analysis`: Generates a list of small multiples each containing the X Chart portion of an XmR Chart from the provided list of DataFrames. For an example of how to use `network_analysis` see the essay [Network Analysis: Advancing the utility of SPC](https://static1.squarespace.com/static/5b722db6f2e6b1ad5053391b/t/679910513be40134de9b54f7/1738084433790/Network+analysis.pdf)
2. `limit_chart_network_analysis`: Generates a grid of small multiples composed of the list of dataframes provided by df_list.

## Notes 
If you are unfamiliar with process behavior charts (control charts) visit  [BrokenQuality.com](https://www.brokenquality.com/). 

## Contributing
To contribute to DataDrivenImprovement, follow these steps:
1. Fork this repository.
2. Create a branch: ```git checkout -b <branch_name>```. 
3. Make your changes and commit them:  ```git commit -m '<commit_message>'```
4. Push to the original branch: ```git push origin <DataDrivenImprovement>/<location>```.
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/creating-a-pull-request). 
## Contact
If you want to contact me you can reach me at [James.Lehner@gmail.com](James.Lehner@gmail.com).
## License
This project uses the following license: MIT License.
## Additional Information
- **Parts of a Process Behavior Chart**: Invented by Dr. Walter Shewhart in the mid-1920s at Bell Laboratories, PBCs are composed of two charts: the `X-chart` and the `mR-chart`. Where the `X-chart` bounds the variation associated with individual values the `mR-chart` bounds the value-to-value variation. This is made possible through the calculation of a trio of limits known as process limits. The `upper process limit (UPL)` and `lower process limit (LPL)` are used on the `X-chart`. The `upper range limit (URL)` is used on the `mR-chart`. 
- **Two types of variation**: Inherent in the characterizations of `predictable` and `unpredictable` is the tyoe of variation action a process. A predictable process is influenced by only `routine` causes of variation. An `unpredictable` process is influenced by both `routine causes of variation` and `assignable` causes of variation.  
- **Improvement**: 
	- **Predictable**: To improve a predictable process `routine` causes of variation must be `identified`, `understood`, and `mitigated`.  This requires fundamental changes to the process must be made. These include, but are not limited to, changes to raw materials, adjustment to system settings, redesign of stations, redesign of software, calibration of measurement systems. 
	- **Unpredictable**: To improve an unpredictable process  `assignable` causes of variation must be `identifed`, `understood`, and `eliminated`. To begin this process, an investigation into values that fall outside the process limits on the `PBC` must be performed. 
- For those unfamiliar with process behavior charts (control charts) that are interested in learning more visit [BrokenQuality.com](https://www.brokenquality.com).
