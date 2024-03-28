## Human in the Loop for Decision-Making
To navigate the intricacies of multiple failure modes, our methodology integrates a human-in-the-loop component. This aspect is crucial for prioritizing and deciding on the order in which to address different failure modes. By leveraging human expertise, we can make informed decisions about which failures to fine-tune first, based on their potential impact, interdependencies, and the overall goals of the system. As given in the following image (or Fig 10. (Alexnet) Appendix) the pretrained had two failure modes but fine tuning on the sample chosen by a human reduced both failure modes.

<p align="center">
  <img src="../images/figure5.png" alt="Probability distribution of actions for AlexNet">
  <br>
  <em>Figure 1: Probability distribution of actions for AlexNet</em>
</p>

## Iterative Loop of Fine-Tuning and Reevaluation
Our iterative loop of fine-tuning and reevaluation, the human-in-the-loop approach enables a responsive and dynamic strategy for managing multiple failure modes. This iterative process, requires human judgment and expertise through which it facilitates a holistic and effective approach to system refinement. It ensures that our efforts to mitigate one failure mode do not inadvertently exacerbate another, aiming for a balanced and comprehensive enhancement of the system. An example image is given below (or Fig 24. Appendix)

<p align="center">
  <img src="../images/figure6.png" alt="Iterative probability shifts in T5 model">
  <br>
  <em>Figure 2: Iterative probability shifts in T5 model</em>
</p>

## Catastrophic forgetting and Selective Application of Fine-Tuning
Catastrophic forgetting where a model loses previously learned information upon learning new information is a critical concern in iterative fine-tuning approaches like ours. To mitigate this issue, we have adopted a specific strategy, detailed further in our methodology.

As part of our fine-tuning protocol, we strategically apply the action derived from the fine-tuning process to the data only 50% of the time. This approach is designed to balance the introduction of new learning with the retention of previously acquired knowledge. By not applying the fine-tuning action universally across all data, we reduce the risk of the model completely "forgetting" its earlier learning due to the overpowering influence of the new data or adjustments. For a detailed explanation of this technique and its underlying rationale, we direct readers to the Appendix where we elaborate on the experimental setup and the specific parameters used.
