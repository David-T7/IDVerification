from django.apps import AppConfig
import threading


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        def _warm_up():
            try:
                from core.utils import warm_up_liveness_models
                warm_up_liveness_models()
            except Exception as exc:
                print(f'Liveness model warm-up failed: {exc}')

        threading.Thread(target=_warm_up, daemon=True).start()
