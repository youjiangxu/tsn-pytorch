import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 timesteps=3, new_length=1, modality='RGB', test_segments=1,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 force_grayscale=False, random_shift=True, test_mode=False, sampling_method='tsn', reverse=False):

        self.root_path = root_path
        self.list_file = list_file
        self.timesteps = timesteps
        self.new_length = new_length
        self.modality = modality
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.test_segments = test_segments
        self.reverse=reverse
        self.sampling_method = sampling_method

        if self.modality == 'RGBDiff':
            self.new_length += 1# Diff needs one more image to calculate diff

        self._parse_list()

    def _load_image(self, directory, idx):
        if self.modality == 'RGB' or self.modality == 'RGBDiff':
            return [Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format(idx))).convert('RGB')]
        elif self.modality == 'Flow':
            x_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('x', idx))).convert('L')
            y_img = Image.open(os.path.join(self.root_path, directory, self.image_tmpl.format('y', idx))).convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file)]

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        if self.sampling_method == 'tsn':
            average_duration = (record.num_frames - self.new_length + 1) // self.timesteps
            if average_duration > 0:
                offsets = np.multiply(list(range(self.timesteps)), average_duration) + randint(average_duration, size=self.timesteps)
            elif record.num_frames > self.timesteps:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.timesteps))
            else:
                offsets = np.zeros((self.timesteps,))
            return offsets + 1
        elif self.sampling_method == 'random':
            if record.num_frames > self.timesteps:
                offsets = np.random.randint(record.num_frames, size=self.timesteps)
            else:
                offsets = np.zeros((self.timesteps,))
            if self.reverse and np.random.randint(1000) > 500:
                offsets[::-1].sort()
            else:
                offsets.sort()
            return np.asarray(offsets)+1
        elif self.sampling_method == 'reverse':
            average_duration = (record.num_frames - self.new_length + 1) // self.timesteps
            if average_duration > 0:
                offsets = np.multiply(list(range(self.timesteps)), average_duration) + randint(average_duration, size=self.timesteps)
            elif record.num_frames > self.timesteps:
                offsets = np.sort(randint(record.num_frames - self.new_length + 1, size=self.timesteps))
            else:
                offsets = np.zeros((self.timesteps,))
            if self.reverse and np.random.randint(1000) > 500:
                offsets = offsets[::-1]
        elif self.sampling_method == 'step':
            total_frame = 40
            assert self.timesteps == 10
            average_duration = (record.num_frames - self.new_length + 1) // total_frame
            step = np.random.randint(1,5)
            offsets = np.multiply(list(range(self.timesteps)), step)
            # print(offsets)

            return offsets + 1

        


    def _get_val_indices(self, record):
        if record.num_frames > self.timesteps + self.new_length - 1:
            if self.sampling_method == 'step':
                total_frame = 40
                assert self.timesteps == 10
                average_duration = (record.num_frames - self.new_length + 1) // total_frame
                step = np.random.randint(1,5)
                offsets = np.multiply(list(range(self.timesteps)), step)
                # print(offsets)

            else:

                tick = (record.num_frames - self.new_length + 1) / float(self.timesteps)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.timesteps)])
        else:
            offsets = np.zeros((self.timesteps,))

        return offsets + 1

    def _get_test_indices(self, record):
        if self.test_segments == 1:
            tick = (record.num_frames - self.new_length + 1) / float(self.timesteps)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.timesteps)])

            return offsets + 1
        elif self.test_segments == 25:
            if self.sampling_method == 'step':
                img_index = []
                total_frame = 35
                assert self.timesteps == 10
                inter_step = (record.num_frames - self.new_length + 1) // total_frame
                for si in range(self.test_segments):
                    for i in range(self.timesteps):
                        img_index.append((si+i)*inter_step)

            else:
                img_index = []
                inter_step = (record.num_frames - self.new_length - self.test_segments + 1) /float(self.timesteps)
                for si in range(self.test_segments):
                    for i in range(self.timesteps):
                        img_index.append(si+i*inter_step)

            print(img_index)
            return np.array(img_index)+1


    def __getitem__(self, index):
        record = self.video_list[index]

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)

        return self.get(record, segment_indices)

    def get(self, record, indices):

        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            for i in range(self.new_length):
                seg_imgs = self._load_image(record.path, p)
                images.extend(seg_imgs)
                if p < record.num_frames:
                    p += 1

        #print('image,', len(images))
        #print('image 0', images[0])
        process_data = self.transform(images)
        #print('process_data,', process_data.size())
        return process_data, record.label

    def __len__(self):
        return len(self.video_list)
