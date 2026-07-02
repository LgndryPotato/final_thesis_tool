<h1>ONNX Backdoor Injection Framework</h1> 

<h2>Tree for the tool</h2>

```text
src/
└── onnx_backdoor/
    ├── __init__.py
    ├── __main__.py                Entry point (allows running via `python -m onnx_backdoor`)
    │
    ├── cli/                       [Component: CLI / API Interface]
    │   ├── __init__.py
    │   ├── config_parser.py       Parses YAML/JSON configs for pipeline execution
    │   └── interface.py           Click/Typer/Argparse logic for user commands
    │
    ├── core/                      [Component: Transformation Engine]
    │   ├── __init__.py
    │   ├── engine.py              Orchestrates the passes and fires events
    │   └── graph_wrapper.py       Utility class holding the in-memory graph state
    │
    ├── io/                        [Component: Model Parser & Serializer]
    │   ├── __init__.py
    │   ├── parser.py              Loads .onnx into an internal graph object
    │   └── serializer.py          Validates and saves modified graph back to .onnx
    │
    └── library/                   [Component: Transformation & Payload Library]
        ├── __init__.py
        ├── base.py                Abstract base class (interface) for all passes
        ├── optimizations/         Standard ML passes (pruning, fusion, etc.)
        │   ├── __init__.py
        │   └── node_fusion.py
        └── payloads/              Adversarial passes (backdoor injections)
            ├── __init__.py
            ├── badnet_injector.py
            └── trigger_subgraph.py
```

<h2>System architecture</h2>

Below is the C4 Component Diagram mapping out the auditable ONNX backdoor injection framework.

![Architecture Diagram](docs/architecture.png)