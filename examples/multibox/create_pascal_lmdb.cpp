#include <opencv2/core/core.hpp>
#include <glog/logging.h>
#include <iostream>
#include <string>

#include "boost/scoped_ptr.hpp"
#include "caffe/caffe.hpp"
#include "caffe/util/db.hpp"

DEFINE_string(label_filename, "", "The file that contains the PASCAL label annotations.");
DEFINE_string(lmdb_dir, "", "The directory that contains the lmdb files.");

using std::string;
using caffe::Datum;
using boost::scoped_ptr;

int main(int argc, char** argv) {
    caffe::GlobalInit(&argc, &argv);

    CHECK(!FLAGS_label_filename.empty()) << "Please set --label_filename.";
    CHECK(!FLAGS_lmdb_dir.empty()) << "Please set --lmdb_dir.";

    LOG(INFO) << "Opening file: " << FLAGS_label_filename;
    std::ifstream label_file(FLAGS_label_filename);

    string image_filename;

    scoped_ptr<caffe::db::DB> db(caffe::db::GetDB("lmdb"));
    db->Open(FLAGS_lmdb_dir, caffe::db::NEW);
    scoped_ptr<caffe::db::Transaction> txn(db->NewTransaction());


    int count =0;
    while (label_file >> image_filename) {
        Datum data;

        // Read the image as a cv::Mat. The 3 channels are in the order or BGR.
        static const bool kIsColor = true;
        cv::Mat image = caffe::ReadImageToCVMat(image_filename, kIsColor);

        // Store the image in a Datum.
        caffe::CVMatToDatum(image, &data);

        int width, height;
        int num_objects;
        CHECK(label_file >> width);
        CHECK(label_file >> height);
        CHECK(label_file >> num_objects);

        data.set_channels(3);
        data.set_width(width);
        data.set_height(height);

        // Ensure that the number of pixel values loaded is correct.
        int num_pixels = 3 * width * height;
        CHECK_EQ(num_pixels, data.data().size());

        // The number of values to load for each annotated object.
        // [xmin, ymin, xmax, ymax, class_id].
        static const int kNumValuesPerObject = 5;
        float value;
        for (int i = 0; i < num_objects; ++i) {
            for (int j = 0; j < kNumValuesPerObject; ++j) {
                CHECK(label_file >> value);
                data.add_float_data(value);
            }
        }

        // Put in db
        string record;
        CHECK(data.SerializeToString(&record));
        txn->Put(image_filename, record);

        if (++count % 50 == 0) {
            // Commit db
            txn->Commit();
            txn.reset(db->NewTransaction());
            LOG(INFO) << "Processed " << count << " files.";
        }
    }
    // Write the last batch
    if (count % 50 != 0) {
        txn->Commit();
        LOG(INFO) << "Processed " << count << " files.";
    }
    return 0;
}



