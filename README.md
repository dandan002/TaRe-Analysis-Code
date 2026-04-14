## Analysis of Tantalum Rhenium Alloys for Use in Superconducting Qubits
This is the companion repository for the thesis with the same title submitted to Princeton University on 04/13/2026. This includes basic SEM images in the SEM directory, a full XPS analysis suite in tare_analysis, the full thesis .tex build in thesis_document, and a phased-buildout of an XRD_analysis pipeline in XRD_analysis, although actual XRD analysis was done through software in provided by the Princeton Imaging and Analysis Center.

In detail:
- The papers directory contains a subset of the literature cited in the thesis.
- tare_analysis contains a full analysis pipeline fashioned from an inherited Tantalum-only analysis pipeline (tool calls from a python notebook, retired due to instability and issues with determinism). Expected Rhenium and Tantalum spectra were added/edited using references from Greiner and other sources. The data used was taken on a ThermoFisher K-Alpha XPS machine.
- XRD_analysis contains a partially-built (unused) XRD analysis pipeline that depends on a local GSAS-II installation in addition to a .env file with a materials project API key (free!). While it turned into an unusable debacle of a codebase, further development is encouraged if only as a lesson in bad coding practices. The data referenced was taken on a Bruker D8 Discover XRD, but final analysis was done on accompanying software provided by Princeton's Imaging and Analysis Center.
- SEM contains scanning electron microscope images taken on a Verios XHR Low-Voltage SEM.
- thesis_document is the full thesis in .tex form with accompanying figures and dependencies copied over from other parts of the repository.

## Usage
Data should be put in tare_analysis/data exported as an .xlsx file from any ThermoFisher K-Alpha or Nexsa series XPS machine; all analysis is orchestrated from the main.py file for surface scans, etch_main.py for etched scans.

## Disclosures
While most of the code was written and/or formatted by hand, figure styling and python tests were written with the assistance of Anthropic's Claude Sonnet 4.5/4.6 models. Generative suggestions were also taken into account while writing XRD_Analysis, but results and the entire codebase were omitted in the final research in favor of analysis done through provided tools. No GenAI or similar tool was used during the writing of the actual document.

The author acknowledges the use of the Imaging and Analysis Center (IAC) operated by the Princeton Materials Institute at Princeton University, which is supported in part by the Princeton Center for Complex Materials (PCCM), a National Science Foundation (NSF) Materials Research Science and Engineering Center (MRSEC; DMR-2011750).