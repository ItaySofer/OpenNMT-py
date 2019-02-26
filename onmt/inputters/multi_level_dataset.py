from onmt import inputters


class MultiLevelDataset(inputters.Dataset):
    """
    A single language tweeter dataset.
    """

    def __init__(self, fields, readers, data, dirs, sort_key, level,
                 filter_pred=None):
        super(MultiLevelDataset, self).__init__(fields, readers, data, dirs, sort_key, filter_pred)
        self.level = level

    def __getitem__(self, idx):
        example = super(MultiLevelDataset, self).__getitem__(idx)
        example.levels = self.level
        return example

    def __iter__(self):
        for i in range(len(self)):
            yield self.__getitem__(i)
