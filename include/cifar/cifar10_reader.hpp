//=======================================================================
// Copyright (c) 2017 Baptiste Wicht
// Distributed under the terms of the MIT License.
// (See accompanying file LICENSE or copy at
//  http://opensource.org/licenses/MIT)
//=======================================================================

/*!
 * \file
 * \brief Contains functions to read the CIFAR-10 dataset
 */

#ifndef CIFAR10_READER_HPP
#define CIFAR10_READER_HPP

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>

namespace cifar {

/*!
 * \brief Represents a complete CIFAR10 dataset
 * \tparam Container The container to use
 * \tparam Image The type of image
 * \tparam Label The type of label
 */
template <template <typename...> class Container, typename Image, typename Label>
struct CIFAR10_dataset {
    Container<Image> training_images; ///< The training images
    Container<Image> test_images;     ///< The test images
    Container<Label> training_labels; ///< The training labels
    Container<Label> test_labels;     ///< The test labels

    /*!
     * \brief Resize the training set to new_size
     *
     * If new_size is less than the current size, this function has no effect.
     *
     * \param new_size The size to resize the training sets to.
     */
    void resize_training(std::size_t new_size) {
        if (training_images.size() > new_size) {
            training_images.resize(new_size);
            training_labels.resize(new_size);
        }
    }

    /*!
     * \brief Resize the test set to new_size
     *
     * If new_size is less than the current size, this function has no effect.
     *
     * \param new_size The size to resize the test sets to.
     */
    void resize_test(std::size_t new_size) {
        if (test_images.size() > new_size) {
            test_images.resize(new_size);
            test_labels.resize(new_size);
        }
    }
};

/*!
 * \brief Read a CIFAR 10 data file inside the given containers
 * \param images The container to fill with the labels
 * \param path The path to the label file
 * \param limit The maximum number of elements to read (0: no limit)
 */
template <typename Images, typename Labels, typename Func>
void read_cifar10_file(Images& images, Labels& labels, const std::string& path, std::size_t limit, Func func) {
    std::ifstream file;
    file.open(path, std::ios::in | std::ios::binary | std::ios::ate);

    if (!file) {
        std::cout << "Error opening file" << std::endl;
        return;
    }

    auto file_size = file.tellg();
    std::unique_ptr<char[]> buffer(new char[file_size]);

    //Read the entire file at once
    file.seekg(0, std::ios::beg);
    file.read(buffer.get(), file_size);
    file.close();

    std::size_t start = images.size();

    size_t size = 10000;

    if(limit > 0 && limit < size){
        size = limit;
    }

    // Prepare the size for the new
    images.reserve(images.size() + size);
    labels.resize(labels.size() + size);

    for(std::size_t i = 0; i < size; ++i){
        labels[start + i] = buffer[i * 3072];

        images.push_back(func());

        for(std::size_t j = 0; i < 3072; ++i){
            images[start + i] = buffer[i * 3072 + j];
        }
    }
}

/*!
 * \brief Read all training data
 *
 * The dataset is assumed to be in a cifar-10 subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 * \param func The functor to create the image objects.
 * \return Container filled with the images
 */
template <typename Images, typename Labels, typename Functor>
void read_training(std::size_t limit, Images& images, Labels& labels, Functor func) {
    read_cifar10_file(images, labels, "cifar-10/cifar-10-batches-bin/data_batch_1.bin", limit, func);
    read_cifar10_file(images, labels, "cifar-10/cifar-10-batches-bin/data_batch_2.bin", limit, func);
    read_cifar10_file(images, labels, "cifar-10/cifar-10-batches-bin/data_batch_3.bin", limit, func);
    read_cifar10_file(images, labels, "cifar-10/cifar-10-batches-bin/data_batch_4.bin", limit, func);
    read_cifar10_file(images, labels, "cifar-10/cifar-10-batches-bin/data_batch_5.bin", limit, func);
}

/*!
 * \brief Read all test images and return a container filled with the images.
 *
 * The dataset is assumed to be in a cifar-10 subfolder
 *
 * \param limit The maximum number of elements to read (0: no limit)
 * \param func The functor to create the image objects.
 * \return Container filled with the images
 */
template <typename Images, typename Labels, typename Functor>
void read_test(std::size_t limit, Images& images, Labels& labels, Functor func) {
    read_cifar10_file(images, labels, "cifar-10/cifar-10-batches-bin/test_batch.bin", limit, func);
}

/*!
 * \brief Read dataset and assume images in 3D (3x32x32)
 *
 * The dataset is assumed to be in a cifar-10 subfolder
 *
 * \param training_limit The maximum number of elements to read from data set (0: no limit)
 *
 * \return The dataset
 */
template <template <typename...> class Container, typename Image, typename Label = uint8_t>
CIFAR10_dataset<Container, Image, Label> read_dataset_3d(std::size_t training_limit = 0) {
    CIFAR10_dataset<Container, Image, Label> dataset;

    read_training<Container, Image>(training_limit, dataset.training_images, dataset.training_labels, [] { return Image(3, 32, 32); });
    read_test<Container, Image>(training_limit, dataset.training_images, dataset.training_labels, [] { return Image(3, 32, 32); });

    return dataset;
}

/*!
 * \brief Read dataset.
 *
 * The dataset is assumed to be in a cifar-10 subfolder
 *
 * \param training_limit The maximum number of elements to read from training set (0: no limit)
 * \param test_limit The maximum number of elements to read from test set (0: no limit)
 * \return The dataset
 */
template <template <typename...> class Container, typename Image, typename Label = uint8_t>
CIFAR10_dataset<Container, Image, Label> read_dataset_direct(std::size_t training_limit = 0, std::size_t test_limit = 0) {
    CIFAR10_dataset<Container, Image, Label> dataset;

    read_training<Container, Image>(training_limit, dataset.training_images, dataset.training_labels, [] { return Image(3 * 32 * 32); });
    read_test<Container, Image>(training_limit, dataset.test_images, dataset.test_labels, [] { return Image(3 * 32 * 32); });

    return dataset;
}

/*!
 * \brief Read dataset.
 *
 * The dataset is assumed to be in a cifar-10 subfolder
 *
 * \param training_limit The maximum number of elements to read from training set (0: no limit)
 * \param test_limit The maximum number of elements to read from test set (0: no limit)
 * \return The dataset
 */
template <template <typename...> class Container = std::vector, template <typename...> class Sub = std::vector, typename Pixel = uint8_t, typename Label = uint8_t>
CIFAR10_dataset<Container, Sub<Pixel>, Label> read_dataset(std::size_t training_limit = 0, std::size_t test_limit = 0) {
    return read_dataset_direct<Container, Sub<Pixel>>(training_limit, test_limit);
}

} //end of namespace cifar

#endif
