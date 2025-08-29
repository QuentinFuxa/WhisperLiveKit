import logging
import importlib.resources as resources
import base64

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
        # Load HTML template
        with resources.files('whisperlivekit.web').joinpath('live_transcription.html').open('r', encoding='utf-8') as f:
            html_content = f.read()
        
        # Load CSS and embed it
        with resources.files('whisperlivekit.web').joinpath('live_transcription.css').open('r', encoding='utf-8') as f:
            css_content = f.read()
        
        # Load JS and embed it
        with resources.files('whisperlivekit.web').joinpath('live_transcription.js').open('r', encoding='utf-8') as f:
            js_content = f.read()
        
        # Load SVG files and convert to data URIs
        with resources.files('whisperlivekit.web').joinpath('src', 'system_mode.svg').open('r', encoding='utf-8') as f:
            system_svg = f.read()
            system_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(system_svg.encode('utf-8')).decode('utf-8')}"
        
        with resources.files('whisperlivekit.web').joinpath('src', 'light_mode.svg').open('r', encoding='utf-8') as f:
            light_svg = f.read()
            light_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(light_svg.encode('utf-8')).decode('utf-8')}"
        
        with resources.files('whisperlivekit.web').joinpath('src', 'dark_mode.svg').open('r', encoding='utf-8') as f:
            dark_svg = f.read()
            dark_data_uri = f"data:image/svg+xml;base64,{base64.b64encode(dark_svg.encode('utf-8')).decode('utf-8')}"
        
        # Replace external references with embedded content
        html_content = html_content.replace(
            '<link rel="stylesheet" href="/web/live_transcription.css" />',
            f'<style>\n{css_content}\n</style>'
        )
        
        html_content = html_content.replace(
            '<script src="/web/live_transcription.js"></script>',
            f'<script>\n{js_content}\n</script>'
        )
        
        # Replace SVG references with data URIs
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
        
        return html_content
        
    except Exception as e:
        logger.error(f"Error creating embedded web interface: {e}")
        return "<html><body><h1>Error loading embedded interface</h1></body></html>"


if __name__ == '__main__':
    
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse
    import uvicorn
    from starlette.staticfiles import StaticFiles
    import pathlib
    import whisperlivekit.web as webpkg
    
    app = FastAPI()    
    web_dir = pathlib.Path(webpkg.__file__).parent
    app.mount("/web", StaticFiles(directory=str(web_dir)), name="web")
    
    @app.get("/")
    async def get():
        return HTMLResponse(get_inline_ui_html())

    uvicorn.run(app=app)
