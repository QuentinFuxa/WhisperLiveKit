class PCMForwarder extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input[0] && input[0].length) {
      // Forward mono channel (0). If multi-channel, downmixing can be added here.
      const channelData = input[0];
      const copy = new Float32Array(channelData.length);
      copy.set(channelData);
      this.port.postMessage(copy, [copy.buffer]);
    }
    // Keep processor alive
    return true;
  }
}

registerProcessor('pcm-forwarder', PCMForwarder);
