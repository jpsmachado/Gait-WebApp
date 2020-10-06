var dictionaryVGG19 = {
    0: 'block1_conv1',
    1: 'block1_conv2',
    2: 'block1_pool',
    3: 'block2_conv1',
    4: 'block2_conv2',
    5: 'block2_pool',
    6: 'block3_conv1',
    7: 'block3_conv2',
    8: 'block3_conv3',
    9: 'block3_conv4',
    10: 'block3_pool',
    11: 'block4_conv1',
    12: 'block4_conv2',
    13: 'block4_conv3',
    14: 'block4_conv4',
    15: 'block4_pool',
    16: 'block5_conv1',
    17: 'block5_conv2',
    18: 'block5_conv3',
    19: 'block5_conv4',
    20: 'block5_pool'
}


function getChannels() {
    var l = window.document.getElementById("layers");
    var layers = l.options[l.selectedIndex].value;
    console.log(String(layers))
    var layername = dictionaryVGG19[layers];
    var channels = 0;
    if (layers < 3) {
        channels = 64;
    } else if (layers >= 3 && layers < 6) {
        channels = 128;
    } else if (layers >= 6 && layers < 11) {
        channels = 256;
    } else {
        channels = 512;
    }
    alert("This layer is a " + layername + ". It has " + String(channels) + " channels.");
}

function getChannels2() {
    var l = window.document.getElementById("layers2");
    var layers = l.options[l.selectedIndex].value;
    console.log(String(layers))
    var layername = dictionaryVGG19[layers];
    var channels = 0;
    if (layers < 3) {
        channels = 64;
    } else if (layers >= 3 && layers < 6) {
        channels = 128;
    } else if (layers >= 6 && layers < 11) {
        channels = 256;
    } else {
        channels = 512;
    }
    alert("This layer is a " + layername + ". It has " + String(channels) + " channels.");
}