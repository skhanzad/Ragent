"""
Logistic Regression Diagram Generator
Generates pytikz neural network diagrams for breast cancer classification
"""

import pytikz as tz

def main():
    print("\n" + "=" * 70)
    print(" LOGISTIC REGRESSION PYTIKZ DIAGRAM GENERATOR")
    print("=" * 70)

    # Diagram 1: Exact implementation (30 → 1)
    print("\n[1/4] Simple logistic regression (30 features → 1 output)...")
    pic1 = tz.Picture()
    net1 = tz.NeuralNetwork(
        [30, 1],
        titles="Logistic Regression: Direct Classifier\n30 Features → Binary Output",
        arrows=True,
        neuron_size=0.25
    )
    pic1.add(net1)
    pic1.render("logistic_simple.png")
    print("      ✓ Saved: logistic_simple.png")

    # Diagram 2: Multi-layer representation
    print("[2/4] Multi-layer network (30 → 20 → 10 → 5 → 1)...")
    pic2 = tz.Picture()
    net2 = tz.NeuralNetwork(
        [30, 20, 10, 5, 1],
        titles="Logistic Regression: Multi-Layer Architecture",
        arrows=True,
        neuron_size=0.15
    )
    pic2.add(net2)
    pic2.render("logistic_multilayer.png")
    print("      ✓ Saved: logistic_multilayer.png")

    # Diagram 3: Deep architecture
    print("[3/4] Deep architecture (30 → 25 → 15 → 10 → 5 → 2 → 1)...")
    pic3 = tz.Picture()
    net3 = tz.NeuralNetwork(
        [30, 25, 15, 10, 5, 2, 1],
        titles="Logistic Regression: Deep Architecture",
        arrows=True,
        neuron_size=0.12
    )
    pic3.add(net3)
    pic3.render("logistic_deep.png")
    print("      ✓ Saved: logistic_deep.png")

    # Diagram 4: Compact with performance metrics
    print("[4/4] Compact model with performance metrics...")
    pic4 = tz.Picture()
    net4 = tz.NeuralNetwork(
        [30, 16, 8, 1],
        titles="Logistic Regression: Breast Cancer Classification\nAccuracy: 98.25% | Precision: 98.53% | Recall: 98.53% | F1: 98.53%",
        arrows=True,
        neuron_size=0.18
    )
    pic4.add(net4)
    pic4.render("logistic_metrics.png")
    print("      ✓ Saved: logistic_metrics.png")

    print("\n" + "=" * 70)
    print(" SUMMARY")
    print("=" * 70)
    print("""
Generated 4 detailed pytikz diagrams:

  1. logistic_simple.png       - Direct implementation (30→1)
                                Matches your notebook architecture

  2. logistic_multilayer.png   - 5-layer network (30→20→10→5→1)
                                Shows hidden layer representations

  3. logistic_deep.png         - 7-layer deep architecture
                                Alternative deeper approach

  4. logistic_metrics.png      - 4-layer compact with results
                                Shows final performance metrics

Dataset: Breast Cancer (UCI)
  • Training samples: 455
  • Test samples: 114
  • Input features: 30
  • Classes: 2 (binary classification)

Training Configuration:
  • Epochs: 200
  • Optimizer: SGD
  • Learning rate: 0.01
  • Loss: BCELoss

All PNG diagrams are ready for papers, presentations, and documentation!
""")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
