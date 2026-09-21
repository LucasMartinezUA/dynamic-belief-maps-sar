# C4 gate

{
  "gate": "NO_DIFFERENTIAL_ADVANTAGE",
  "checks": {
    "online_static_3step": {
      "1": [
        "INCONCLUSIVE",
        "INCONCLUSIVE",
        "WORSENING"
      ],
      "2": [
        "WORSENING",
        "INCONCLUSIVE",
        "WORSENING"
      ],
      "3": [
        "WORSENING",
        "WORSENING",
        "WORSENING"
      ]
    },
    "pizza_repartition": {
      "1": [
        "INCONCLUSIVE",
        "INCONCLUSIVE",
        "INCONCLUSIVE"
      ],
      "2": [
        "INCONCLUSIVE",
        "INCONCLUSIVE",
        "INCONCLUSIVE"
      ],
      "3": [
        "IMPROVEMENT",
        "INCONCLUSIVE",
        "INCONCLUSIVE"
      ]
    }
  },
  "online_pizza_states": {
    "1": [
      "INCONCLUSIVE",
      "INCONCLUSIVE",
      "IMPROVEMENT"
    ],
    "2": [
      "INCONCLUSIVE",
      "INCONCLUSIVE",
      "IMPROVEMENT"
    ],
    "3": [
      "IMPROVEMENT",
      "IMPROVEMENT",
      "IMPROVEMENT"
    ]
  },
  "dynamic_online_equivalent": false,
  "online_pizza_improved": false,
  "no_worsening_all": false,
  "fairness_valid": true,
  "absolute_endpoints_secondary": true,
  "wording_prohibition": "do not call this fault tolerance"
}
