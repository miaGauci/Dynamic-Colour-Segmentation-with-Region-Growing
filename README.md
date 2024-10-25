# Dynamic-Colour-Segmentation-with-Region-Growing

Developing a computer-vision algorithm that identifies and extracts regions of similar colour within an image. By allowing users to select a seed pixel, the algorithm highlights the predominant colour in the selected region for easier recognition and analysis.

- The user is prompted to select an image and seed pixel. The algorithm starts by retrieving the RGB values of the seed pixel and calls functions to identify the region of a similar colour.
- Using a queue-based breadth-first search, the algorithm expands from the seed pixel to neighbouring pixels within a user-defined threshold. A post-processing step fills in gaps using imfill and dilation.
- The threshold parameter adjusts the colour similarity tolerance, allowing more accurate region identification based on the seed pixel.
- *getColourName* categorises colours based on RGB thresholds. Adjustments were made through testing to optimise the classification.
- The *evaluateSegmentation* function calculates accuracy, precision, and sensitivity, comparing the algorithm's segmentation with a user-created ground truth mask.
- The Euclidean Distance formula measures the difference between the seed pixel and the average region colour, giving insight into the colour variation within the region.
