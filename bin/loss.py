from keras import backend

def content_loss(content, result):
    return backend.sum(backend.square(result - content))

def gram_matrix(x):
    features = backend.batch_flatten(backend.permute_dimensions(x, (2, 0, 1)))
    gram = backend.dot(features, backend.transpose(features))
    return gram

def style_loss(style, result, h, w):
    s = gram_matrix(style)
    r = gram_matrix(result)
    channels = 3
    return backend.sum(backend.square(s - r)) / float((2 * h * w * channels) ** 2)

def total_loss(x, h, w):
    val_one = backend.square(x[:, :h-1, :w-1, :] - x[:, 1:, :w-1, :])
    val_two = backend.square(x[:, :h-1, :w-1, :] - x[:, :h-1, 1:, :])
    return backend.sum(backend.pow(val_one + val_two, 1.25))