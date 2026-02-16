# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for agos CLI — single-file Windows executable."""

block_cipher = None

a = Analysis(
    ['agos_entry.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        # CLI
        'agos.cli.main',
        'agos.cli.agents',
        'agos.cli.system',
        'agos.cli.context',
        'agos.cli.intent',
        # Kernel
        'agos.kernel.agent',
        'agos.kernel.runtime',
        'agos.kernel.state_machine',
        # LLM
        'agos.llm.anthropic',
        'agos.llm.base',
        # Tools
        'agos.tools.builtins',
        'agos.tools.registry',
        'agos.tools.schema',
        # Knowledge
        'agos.knowledge.manager',
        'agos.knowledge.base',
        'agos.knowledge.episodic',
        'agos.knowledge.semantic',
        'agos.knowledge.graph',
        'agos.knowledge.learner',
        'agos.knowledge.consolidator',
        'agos.knowledge.note',
        'agos.knowledge.working',
        # Intent
        'agos.intent.engine',
        'agos.intent.planner',
        'agos.intent.personas',
        'agos.intent.proactive',
        # Triggers
        'agos.triggers.base',
        'agos.triggers.manager',
        'agos.triggers.schedule',
        'agos.triggers.file_watch',
        'agos.triggers.webhook',
        # Events
        'agos.events.bus',
        'agos.events.tracing',
        # Evolution
        'agos.evolution.scout',
        'agos.evolution.analyzer',
        'agos.evolution.engine',
        'agos.evolution.repo_scout',
        'agos.evolution.code_analyzer',
        'agos.evolution.sandbox',
        'agos.evolution.integrator',
        'agos.evolution.pipeline',
        'agos.evolution.daemon',
        'agos.evolution.strategies.memory_softmax',
        'agos.evolution.strategies.memory_layered',
        'agos.evolution.strategies.memory_semaphore',
        'agos.evolution.strategies.memory_confidence',
        # Policy
        'agos.policy.engine',
        'agos.policy.audit',
        'agos.policy.schema',
        # Coordination
        'agos.coordination.channel',
        'agos.coordination.team',
        'agos.coordination.workspace',
        # Ambient
        'agos.ambient.watcher',
        # Dashboard
        'agos.dashboard.app',
        # Migrations
        'agos.migrations.runner',
        'agos.migrations.m_001_initial',
        # Core modules
        'agos.updater',
        'agos.release',
        # Dependencies
        'aiosqlite',
        'pydantic',
        'pydantic_settings',
        'anthropic',
        'httpx',
        'orjson',
        'structlog',
        'typer',
        'rich',
        'click',
        'packaging',
        'sqlite3',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='agos',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
