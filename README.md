## Tensorflow-based Guitar Amp and Effect Modeler

Based on the research: [Real-Time Guitar Amplifier Emulation with Deep Learning](https://www.mdpi.com/2076-3417/10/3/766)

And heavily influenced by: [PedalNetRT](https://github.com/GuitarML/PedalNetRT)

### Outline

The purpose of this project was to see if I could build the neural network described in the Real-Time Guitar Amplifier Emulation with Deep Learning paper using Tensorflow, and build my own model snapshots of software amplifiers that I use in my own recordings.

The model is supplied a 3-4min clip of DI guitar input and associated output from the amp or effect we're trying to model. For proof of concept I'm modeling software amplifiers, particularly those in [Neural DSP's Archetype: Nolly](https://neuraldsp.com/plugins/archetype-nolly?srsltid=AfmBOooHhgYWDFGuZKiJoEpmGdnre6s0M4bNjbrj8iwtTl4unvHxUrv_) plugin, as this greatly simplifies the training data generation compared to hardware amplifiers. Because of this, we already have the guitar DI and isolating the amplifier output (without pedals, cabinets, microphones, etc.) is as simple as turning off everything else in the chain. As I understand it this can be used to model any non-time-based effects, i.e. preamps, overdrives, compressors, etc.

### Performance and Output
As per my favorite music, my main goal is to model high gain amplifiers. Those are tougher for the model to train than a clean guitar signal, I would assume because the input and output signal are much more different than a DI and clean amp signal. But we push on anyhow.

Here's an example of a death metal tone (1000 Epochs, 20% ESR):
<img width="1268" height="599" alt="EWBHeadSettings" src="https://github.com/user-attachments/assets/30b2a3fb-fcfd-4761-8fbc-9b84bef98fe8" />
(insert audio samples)

It's not perfect, but to my ears it sounds much closer than I'd expect for 20% error. And after plugging it into the full signal chain with a cabinet IR, the difference is even less noticable:

### Future plans:
- There are VST plugins for loading models to use as real-time guitar effects. I may write one myself for giggles, but in the meantime I'm just going to use: [SmartGuitarPedal](https://github.com/GuitarML/SmartGuitarPedal)
- This process produces a model of an amp or effect with static settings, which isn't particularly useful in practice. The Neural DSP folks have written a paper: [End-To-End Amp Modeling: From Data to Controllable Guitar Amplifier Models](https://arxiv.org/pdf/2403.08559) about the capturing the setting changes here: and I aim to implement that as well. Once again aided by my self-isolation in the software realm, the automation of settings will be simple as compared to their knob-turning robot.
