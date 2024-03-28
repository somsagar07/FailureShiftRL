## Analysis with adversarial training

Adversarial training is a method aimed at improving model robustness by explicitly training the model on adversarial examples. These examples are intentionally designed to be close to real data points but tweaked slightly to fool the model into making incorrect predictions or classifications. Our approach seeks to comprehensively map the model’s failure landscape across a wider range of scenarios beyond adversarial vulnerabilities. The goal is to identify both obvious and subtle failure modes, including those not necessarily related to adversarial attacks.

We have conducted an extended study by utilizing Fast Gradient Sign Method (FGSM), an adversarial attack technique, to improve the model’s robustness. We found that while adversarial training techniques like FGSM enhance model robustness, they primarily do so by altering samples close to the decision boundary. This phenomenon is evident from the Figure 1, where points near the original image coordinate (0, 0, 0) exhibit greater resilience to failure compared to those positioned further away. Despite the relative safety of nearby perturbations, we discovered that the model remains susceptible to failures at more distant points, as exemplified by the instance marked with a yellow circle at the coordinate (3, 4, 4).

<p align="center">
  <img src="../images/figure7.png" style="width:50%;" alt="Failure landscape of AlexNet (Adversarially trained on FGSM) with rotation, darkening and saturation actions.">
  <br>
  <em>Figure 1: Failure landscape of AlexNet (Adversarially trained on FGSM) with rotation, darkening and saturation actions.</em>
</p>

We also found that even after adversarial training, when tested against the same attack, the model still exhibits vulnerabilities with some other pertubations as shown in Figure 2. This observation further underscores the importance of summarize step before going for reconstruction of the decision boundary.

<p align="center">
  <img src="../images/figure8.png" style="width:50%;" alt="Failure landscape of AlexNet (Adversarially trained on FGSM) with rotation, darkening and saturation actions.">
  <br>
  <em>Figure 2: Failure landscape of AlexNet (Adversarially trained on FGSM) with rotation, darkening and FGSM actions.</em>
</p>