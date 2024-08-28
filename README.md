# Swiss traffic model for Light Commercial Vehicles (LCV)

This repository hosts the code and documentation relative to the Swiss model for Light Commercial Vehicles, as well as a smaller separate model, which is specific for parcel deliveries.
The LCV model only covers the vehicles owned by juridical persons. Those owned by private persons are modeled in the national passenger transport model (c.f. https://www.are.admin.ch/npvm).

For an overview of the Swiss model for LCVs, see: https://github.com/AREschweiz/LCV_model/blob/main/Swiss_LCV_model_technical_report.pdf

For a description of the model for parcel deliveries, see first the appendix B of the report referenced above. Then for technical details, the Section 4.1.5 of the following report:
https://github.com/AREschweiz/LCV_model/blob/main/22064-R01%20-%20Report%20Audit%20and%20update%20of%20Swiss%20national%20LCV%20model%20-%20update%2020230602.pdf

The trip matrices produced by the LCV model and by the parcel delivery model are assigned to the road network within the national passenger transport model. These matrices are then calibrated against traffic counts. The calibrated matrices for the year 2023 will be made available on Zenodo in 2025 (hopefully Q1).

Note: in order to apply the model, one needs the number of FTE per NOGA Section at the level of the zones of the transport model. This data is not publicly available.
