<!-- Improved compatibility of back to top link: See: https://github.com/othneildrew/Best-README-Template/pull/73 -->

<a id="readme-top"></a>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->

<h3 align="center">Internship Dissertation Project 2024

<!-- ABOUT THE PROJECT -->

## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

The following GitHub repository highlights the work done by me (Sean Lim) during my internship-dissertation program at City, University of London under the company AITIS (AITI Solutions).

In this project, I attempt to improve on the current state of object to object distance estimation by introducing OriDim, a model that estimates an object's dimensions and orientation through input cropped images of various objects detected by an object detection algorithm (YOLO, for example).

The dataset used is a custom dataset provided by AITIS concerning objects in a Construction Site (workers, pump trucks, cranes, forklifts etc.) but is not provided in this repository due to privacy and sensitivity issues. Instead, I have included images taken from a high rise building with similar objects in a Construction Site to showcase the performance of my model.</p>

<!-- GETTING STARTED -->

## Getting Started

### Installation

1. Clone the repo

   ```
   git clone https://github.com/Seanlim107/Internship-Project-2024
   ```
2. Install packages

   ```
   pip install -r requirements.txt
   ```
3. Download checkpoints [here](https://drive.google.com/drive/u/1/folders/10OclwYqrT4uszMrPMUZ6xA7OPZUfyrtg) and put the models in the **checkpoints** folder

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- USAGE EXAMPLES -->

## Usage

The following section discusses the files that can be run to test the performance of the OriDim model and its variants. All test cases can be run simply from Inference v1.py, Inference v2.py and Inference v3.py

The file Inference v1.py showcases the performance of traditional methods being 2D object detection e.g. [Social Distancing Detector](https://pyimagesearch.com/2020/06/01/opencv-social-distancing-detector/) and this [Distance Measuring Tool](https://pyimagesearch.com/2016/04/04/measuring-distance-between-objects-in-an-image-with-opencv/https:/)

The file Inference v2.py tests the performance of the OriDim model on the ACSD dataset, however this file is currently unavailable due to the lack of availability of the ACSD dataset from AITIS due to privacy and security reasons.

The file Inference v3.py instead is an alternative to Inference v2.py which shows the works of the OriDim model on a custom made dataset with pictures taken manually from a high rise building on a construction site in the UK.

Available configurations:


|    | 3d_guesser_classification | 3d_guesser_proposals | 3d_guesser_backbone |
| ---- | :-------------------------- | ---------------------- | --------------------- |
| 1  | True                      | 8                    | efficientnet_b0     |
| 2  | True                      | 32                   | efficientnet_b0     |
| 3  | False                     | -                    | efficientnet_b0     |
| 4  | True                      | 8                    | efficientnet_b2     |
| 5  | True                      | 32                   | efficientnet_b2     |
| 6  | False                     | -                    | efficientnet_b2     |
| 7  | True                      | 8                    | MNASNet1_0          |
| 8  | True                      | 32                   | MNASNet1_0          |
| 9  | False                     | -                    | MNASNet1_0          |
| 10 | True                      | 8                    | MobileNet_V2        |
| 11 | True                      | 16                   | MobileNet_V2        |
| 12 | True                      | 32                   | MobileNet_V2        |
| 13 | False                     | -                    | MobileNet_V2        |


<p align="right">(<a href="#readme-top">back to top</a>)</p>


## <!-- CONTRIBUTING -->

## Contributions

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Top contributors:

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- CONTACT -->

## Contact

Sean Lim  - seanlim1071@gmail.com

Project Link: [https://github.com/Seanlim107/Internship-Project-2024/tree/main/runshttps://github.com/your_username/repo_nam](https://github.com/Seanlim107/Internship-Project-2024/tree/main/runshttps://github.com/your_username/repo_nam)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)
* [Malven's Flexbox Cheatsheet](https://flexbox.malven.co/)
* [Malven's Grid Cheatsheet](https://grid.malven.co/)
* [Img Shields](https://shields.io)
* [GitHub Pages](https://pages.github.com)
* [Font Awesome](https://fontawesome.com)
* [React Icons](https://react-icons.github.io/react-icons/search)

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
