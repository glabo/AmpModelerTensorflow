## Tensorflow-based Guitar Amp and Effect Modeler

Based on the research: [Real-Time Guitar Amplifier Emulation with Deep Learning](https://www.mdpi.com/2076-3417/10/3/766)

And heavily influenced by: [PedalNetRT](https://github.com/GuitarML/PedalNetRT)

### Outline

The purpose of this project was to see if I could build the neural network described in the Real-Time Guitar Amplifier Emulation with Deep Learning paper using Tensorflow, and build my own model snapshots of software amplifiers that I use in my own recordings.

The model is supplied a 3-4min clip of DI guitar input and associated output from the amp or effect we're trying to model. For proof of concept I'm modeling software amplifiers, particularly those in [Neural DSP's Archetype: Nolly](https://neuraldsp.com/plugins/archetype-nolly?srsltid=AfmBOooHhgYWDFGuZKiJoEpmGdnre6s0M4bNjbrj8iwtTl4unvHxUrv_) plugin, as this greatly simplifies the training data generation compared to hardware amplifiers. Because of this, we already have the guitar DI and isolating the amplifier output (without pedals, cabinets, microphones, etc.) is as simple as turning off everything else in the chain. As I understand it this can be used to model any non-time-based effects, i.e. preamps, overdrives, compressors, etc.

### Performance and Examples
In the examples the "raw" tone refers to the signal straight from the amp, without a cabinet or effects. The "full chain" is as it sounds, with effects and cabinet impulses applied. "Expected" output is produced by the plugin that we're modeling.

Here's an example of a clean jazz tone (1000 Epochs, 13% ESR):
<img width="1102" height="530" alt="SingSinginHeadSettings" src="https://github.com/user-attachments/assets/d98a219c-b327-482f-b12e-2fe5143773f6" />
- [DI](https://github.com/user-attachments/assets/157c1bb1-5616-466c-a814-56c595350336)
- [Amp-Only Expected Output](https://github.com/user-attachments/assets/83304969-2ef4-47f5-b896-73d67657e5f1)
- [Amp-Only Modeled Output](https://github.com/user-attachments/assets/44743169-6f85-4f70-8eb9-e8b8d34a7cf9)
- [Full Chain Expected Output](https://github.com/user-attachments/assets/25757873-41b1-4af5-b263-661de9dc5ad2)
- [Full Chain Modeled Output](https://github.com/user-attachments/assets/8b741004-51fd-41ba-bf23-43826bea3db8)

It's not perfect, the most noticable difference is the model is not as bright as the expected output, but it's pretty close and definitely servicable.

As per my favorite music, my main goal is to model high gain amplifiers. Those are tougher for the model to train than a clean guitar signal, I would assume because the input and output signal are much more different than a DI and clean amp signal. But we push on anyhow.

Here's an example of a heavy metal tone (1000 Epochs, 20% ESR):
<img width="1268" height="599" alt="EWBHeadSettings" src="https://github.com/user-attachments/assets/a880032f-93f1-4f4f-b6b7-0a1adada8a09" />

Though it would provide the most scientific comparison, raw high gain amplifier output is gross. In this case the full chain only emphasizes the difference between the two, so we'll only show full-chain output here.
- [Full Chain Expected](https://github.com/user-attachments/assets/d60aa4f0-4d5e-4e00-b169-cd533844d63b)
- [Full Chain Modeled](https://github.com/user-attachments/assets/1062a29e-310f-4473-8cd7-fdab81559f52)

Above ~225Hz these signals are quite similar, but sub-225Hz frequencies in the modeled signal are ~3dB lower than in the expected signal. The difference here is certainly larger than the clean tone, which is to be expected given the ESR. You can see this reflected in a visual comparison of the output signals, the small-and-fast high frequency changes are fairly faithfully followed, but the low frequency voltage-offset-like changes are largely lost. In this chart "y_test" is the expected output and "y_pred" is the generated output from the model.

<img width="1061" height="194" alt="detail_signal_comparison_e2s_0 3417_cropped" src="https://github.com/user-attachments/assets/e90a16bd-4a7a-4eaf-8121-3453be1e3498" />

### Tangential Speculation
I suspect the loss in low end could in-part be the result of the pre-emphasis filtering, which emphasizes higher frequencies as the model struggles to replicate them. There's likely some tuning to be done there, and I have a theory that in practice they use a combination of model and EQ to achieve the desired output.

Ultimately, there's an interesting caveat to the application of producing audio: two waveforms don't have to be identical for our ears to recognize them as quite similar, so you could in theory "get away" with much more error than you would in other ML applications. There is also no inherent practical value in achieving a 1:1 replica of a particular amp as long as you have captured the "character" of said amp; beyond that you're only looking to produce a "good feeling" tool for a guitar player to make noise with. My ears can tell the difference between the above two tones in head-to-head comparison, and the one I've produced certainly requires a bit of working to better cover the frequency spectrum before I'd call it complete, but despite 20% error it sure sounds like a Peavey 5150 to me. All of that to say: I'd be curious to see the acceptable ESR threshhold when companies produce products from this, as I'm sure it's really easy to fixate on achieving low-single-digit ESR when it may take less to fool your ears and even less to achieve recognizable resemblance.

I also think I could get the model to work better and achieve lower ESR.

### Future plans:
- There are VST plugins for loading models to use as real-time guitar effects. I may write one myself for giggles, but in the meantime I'm just going to use: [SmartGuitarPedal](https://github.com/GuitarML/SmartGuitarPedal)
- This process produces a model of an amp or effect with static settings, which isn't particularly useful in practice. The Neural DSP folks have written a paper: [End-To-End Amp Modeling: From Data to Controllable Guitar Amplifier Models](https://arxiv.org/pdf/2403.08559) about the capturing amplifiers with dynamic settings and I aim to implement that as well. Once again aided by my self-isolation in the software realm, the automation of settings will be simple as compared to their knob-turning robot.
