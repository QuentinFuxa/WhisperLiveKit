import base64
import importlib.resources as resources
import logging

logger = logging.getLogger(__name__)

def get_web_interface_html():
    """Loads the HTML for the web interface using importlib.resources."""
    try:
        with resources.files('whisperlivekit.web').joinpath('live_transcription.html').open('r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error loading web interface HTML: {e}")
        return "<html><body><h1>Error loading interface</h1></body></html>"

def get_inline_ui_html():
    """Returns the complete web interface HTML with all assets embedded in a single call."""
    try:
        with resources.files('whisperlivekit.web').joinpath('live_transcription.html').open('r', encoding='utf-8') as f:
            html_content = f.read()        
        with resources.files('whisperlivekit.web').joinpath('live_transcription.css').open('r', encoding='utf-8') as f:
            css_content = f.read()
        with resources.files('whisperlivekit.web').joinpath('live_transcription.js').open('r', encoding='utf-8') as f:
            js_content = f.read()
        
        with resources.files('whisperlivekit.web').joinpath('pcm_worklet.js').open('r', encoding='utf-8') as f:
            worklet_code = f.read()
        with resources.files('whisperlivekit.web').joinpath('recorder_worker.js').open('r', encoding='utf-8') as f:
            worker_code = f.read()
        
        js_content = js_content.replace(
            'await audioContext.audioWorklet.addModule("/web/pcm_worklet.js");',
            'const workletBlob = new Blob([`' + worklet_code + '`], { type: "application/javascript" });\n' +
            'const workletUrl = URL.createObjectURL(workletBlob);\n' +
            'await audioContext.audioWorklet.addModule(workletUrl);'
        )
        js_content = js_content.replace(
            'recorderWorker = new Worker("/web/recorder_worker.js");',
            'const workerBlob = new Blob([`' + worker_code + '`], { type: "application/javascript" });\n' +
            'const workerUrl = URL.createObjectURL(workerBlob);\n' +
            'recorderWorker = new Worker(workerUrl);'
        )
        
        # SVG files
        with resources.files('whisperlivekit.web').joinpath('src', 'system_mode.svg').open('r', encoding='utf-8') as f:
            system_svg = f.read()
            system_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(system_svg.encode('utf-8')).decode('utf-8')}"
        with resources.files('whisperlivekit.web').joinpath('src', 'light_mode.svg').open('r', encoding='utf-8') as f:
            light_svg = f.read()
            light_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(light_svg.encode('utf-8')).decode('utf-8')}"
        with resources.files('whisperlivekit.web').joinpath('src', 'dark_mode.svg').open('r', encoding='utf-8') as f:
            dark_svg = f.read()
            dark_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(dark_svg.encode('utf-8')).decode('utf-8')}"
        with resources.files('whisperlivekit.web').joinpath('src', 'settings.svg').open('r', encoding='utf-8') as f:
            settings = f.read()
            settings_uri = f"data:image/svg+xml;base64,{base64.b64encode(settings.encode('utf-8')).decode('utf-8')}"

        # Replace external references
        html_content = html_content.replace(
            '<link rel="stylesheet" href="live_transcription.css" />',
            f'<style>\n{css_content}\n</style>'
        )
        
        html_content = html_content.replace(
            '<script src="live_transcription.js"></script>',
            f'<script>\n{js_content}\n</script>'
        )
        
        # Replace SVG references
        html_content = html_content.replace(
            '<img src="/web/src/system_mode.svg" alt="" />',
            f'<img src="{system_data_uri}" alt="" />'
        )
        
        html_content = html_content.replace(
            '<img src="/web/src/light_mode.svg" alt="" />',
            f'<img src="{light_data_uri}" alt="" />'
        )
        
        html_content = html_content.replace(
            '<img src="/web/src/dark_mode.svg" alt="" />',
            f'<img src="{dark_data_uri}" alt="" />'
        )
        
        html_content = html_content.replace(
            '<img src="web/src/settings.svg" alt="Settings" />',
            f'<img src="{settings_uri}" alt="" />'
        )
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error creating embedded web interface: {e}")
        return "<html><body><h1>Error loading embedded interface</h1></body></html>"


if __name__ == '__main__':
    
    import pathlib

    import uvicorn
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    from starlette.staticfiles import StaticFiles

    import whisperlivekit.web as webpkg
    
    app = FastAPI()    
    web_dir = pathlib.Path(webpkg.__file__).parent
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")
    
    @app.get("/")
    async def get():
        return HTMLResponse(get_inline_ui_html())

    uvicorn.run(app=app)
