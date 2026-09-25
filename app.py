import gradio as gr
from core import *
from urllib.parse import urlparse, parse_qs
from contextlib import suppress

BASE_DIR = os.getcwd()

output_dir = os.path.join(BASE_DIR, 'audios')

def get_youtube_video_id(url, ignore_playlist=True):
    """Extract YouTube video ID from various URL formats"""
    query = urlparse(url)
    
    # Handle youtu.be URLs
    if query.hostname == 'youtu.be':
        video_id = query.path[1:]
        if video_id:
            return video_id
        return None

    # Handle youtube.com URLs
    if query.hostname in {'www.youtube.com', 'youtube.com', 'music.youtube.com'}:
        if query.path == '/watch':
            video_id = parse_qs(query.query).get('v', [None])[0]
            if video_id:
                return video_id
        elif query.path.startswith('/watch/'):
            return query.path.split('/')[2] if len(query.path.split('/')) > 2 else None
        elif query.path.startswith('/embed/'):
            return query.path.split('/')[2] if len(query.path.split('/')) > 2 else None
        elif query.path.startswith('/v/'):
            return query.path.split('/')[2] if len(query.path.split('/')) > 2 else None
        elif not ignore_playlist:
            # For playlist URLs, extract playlist ID
            playlist_id = parse_qs(query.query).get('list', [None])[0]
            if playlist_id:
                return playlist_id
    
    return None


def yt_download(link, cookie_file=None, cookie_browser=None):
    """Download audio from YouTube"""
    if not link or not link.strip():
        error_msg = 'No URL provided. Please enter a valid YouTube URL.'
        raise_exception(error_msg)
    
    # Validate URL
    if urlparse(link).scheme not in ['http', 'https']:
        error_msg = 'Invalid URL. Please provide a valid YouTube URL.'
        raise_exception(error_msg)
    
    # Check if it's a valid YouTube URL
    song_id = get_youtube_video_id(link)
    if song_id is None:
        error_msg = 'Invalid YouTube URL. Please check the URL and try again.'
        raise_exception(error_msg)

    # Build yt-dlp options with cookie support
    ydl_opts = {
        'format': 'bestaudio',
        'outtmpl': os.path.join(output_dir, '%(title)s.%(ext)s'),
        'nocheckcertificate': True,
        'ignoreerrors': True,
        'no_warnings': True,
        'quiet': True,
        'extractaudio': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    # Priority 1: use uploaded cookie file
    if cookie_file is not None:
        ydl_opts['cookiefile'] = cookie_file.name if hasattr(cookie_file, 'name') else str(cookie_file)
    # Priority 2: extract cookies from a browser
    elif cookie_browser and cookie_browser != 'none':
        ydl_opts['cookiesfrombrowser'] = (cookie_browser,)

    # Common anti-bot settings
    ydl_opts.setdefault('http_headers', {})
    ydl_opts['http_headers']['User-Agent'] = (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/135.0.0.0 Safari/537.36'
    )
    ydl_opts['extractor_args'] = {'youtube': {'player_client': ['ios', 'web']}}
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Download the audio
            ydl.download([link])
            
            # Get the downloaded file info
            info = ydl.extract_info(link, download=False)
            title = info.get('title', 'audio')
            download_path = os.path.join(output_dir, f"{title}.mp3")
            
            # Check if file exists
            if not os.path.exists(download_path):
                # Try alternative naming (sometimes yt-dlp adds suffixes)
                import glob
                files = glob.glob(os.path.join(output_dir, f"{title}*.mp3"))
                if files:
                    download_path = files[0]
                else:
                    error_msg = 'Failed to download or locate the audio file.'
                    raise_exception(error_msg)
            
            return download_path
            
    except gr.Error:
        raise
    except Exception as e:
        error_msg = f'Error downloading from YouTube: {str(e)}'
        if 'Sign in to confirm' in str(e) or 'bot' in str(e).lower():
            error_msg += (
                '\n\nTip: YouTube requires cookies to bypass bot detection. '
                'Try uploading a cookies.txt file or selecting a browser to extract cookies from.'
            )
        raise_exception(error_msg)



with gr.Blocks(title="🔊",theme=gr.themes.Base(primary_hue="blue",neutral_hue="zinc")) as app:
    with gr.Row():
        gr.Markdown("# EasyGUI V3")
    with gr.Tabs():
        with gr.TabItem(i18n("模型推理")):
            with gr.Row():
                sid0 = gr.Dropdown(label=i18n("推理音色"), choices=sorted(names), value=find_model())
                refresh_button = gr.Button(i18n("刷新音色列表和索引路径"), variant="primary")
                #clean_button = gr.Button(i18n("卸载音色省显存"), variant="primary")
                spk_item = gr.Slider(
                    minimum=0,
                    maximum=2333,
                    step=1,
                    label=i18n("请选择说话人id"),
                    value=0,
                    visible=False,
                    interactive=True,
                )

                vc_transform0 = gr.Number(
                    label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                )
                but0 = gr.Button(i18n("转换"), variant="primary")
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        dropbox = gr.Audio(label="Drop your audio here & hit the Reload button.", type="filepath")
                    with gr.Row():
                        record_button=gr.Audio(sources="microphone", label="OR Record audio.", type="filepath")
                    with gr.Row():
                        input_audio0 = gr.Dropdown(
                            label=i18n("输入待处理音频文件路径(默认是正确格式示例)"),
                            value=find_audios(True),
                            choices=find_audios()
                        )
                        record_button.change(fn=save_wav, inputs=[record_button], outputs=[input_audio0])
                        dropbox.upload(fn=save_wav, inputs=[dropbox], outputs=[input_audio0])
                with gr.Column():
                    with gr.Accordion(label=i18n("自动检测index路径,下拉式选择(dropdown)"), open=False):
                        file_index2 = gr.Dropdown(
                            label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                            choices=get_indexes(),
                            interactive=True,
                            value=get_index()
                        )
                        index_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=0.66,
                            interactive=True,
                        )
                    vc_output2 = gr.Audio(label=i18n("输出音频(右下角三个点,点了可以下载)"))
                    with gr.Accordion(label=i18n("常规设置"), open=False):
                        f0method0 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                            ),
                            choices=["pm", "harvest", "crepe", "rmvpe"]
                            if config.dml == False
                            else ["pm", "harvest", "rmvpe"],
                            value="rmvpe",
                            interactive=True,
                        )
                        filter_radius0 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3,
                            step=1,
                            interactive=True,
                        )
                        resample_sr0 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                        rms_mix_rate0 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=0.21,
                            interactive=True,
                        )
                        protect0 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                    file_index1 = gr.Textbox(
                        label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                        value="",
                        interactive=True,
                        visible=False
                    )
                    refresh_button.click(
                        fn=change_choices,
                        inputs=[],
                        outputs=[sid0, file_index2, input_audio0],
                        api_name="infer_refresh",
                    )
                    # file_big_npy1 = gr.Textbox(
                    #     label=i18n("特征文件路径"),
                    #     value="E:\\codes\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                    #     interactive=True,
                    # )
            with gr.Row():
                f0_file = gr.File(label=i18n("F0曲线文件, 可选, 一行一个音高, 代替默认F0及升降调"), visible=False)
            with gr.Row():
                vc_output1 = gr.Textbox(label=i18n("输出信息"))
                but0.click(
                    vc.vc_single,  
                    [
                        spk_item,
                        input_audio0,
                        vc_transform0,
                        f0_file,
                        f0method0,
                        file_index1,
                        file_index2,
                        # file_big_npy1,
                        index_rate1,
                        filter_radius0,
                        resample_sr0,
                        rms_mix_rate0,
                        protect0,
                    ],
                    [vc_output1, vc_output2],
                    api_name="infer_convert",
                )
            with gr.Row():
                with gr.Accordion(open=False, label=i18n("批量转换, 输入待转换音频文件夹, 或上传多个音频文件, 在指定文件夹(默认opt)下输出转换的音频. ")):                
                    with gr.Row():
                        opt_input = gr.Textbox(label=i18n("指定输出文件夹"), value="opt")
                        vc_transform1 = gr.Number(
                            label=i18n("变调(整数, 半音数量, 升八度12降八度-12)"), value=0
                        )
                        f0method1 = gr.Radio(
                            label=i18n(
                                "选择音高提取算法,输入歌声可用pm提速,harvest低音好但巨慢无比,crepe效果好但吃GPU,rmvpe效果最好且微吃GPU"
                            ),
                            choices=["pm", "harvest", "crepe", "rmvpe"]
                            if config.dml == False
                            else ["pm", "harvest", "rmvpe"],
                            value="pm",
                            interactive=True,
                        )
                    with gr.Row():
                        filter_radius1 = gr.Slider(
                            minimum=0,
                            maximum=7,
                            label=i18n(">=3则使用对harvest音高识别的结果使用中值滤波，数值为滤波半径，使用可以削弱哑音"),
                            value=3,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                    with gr.Row():
                        file_index3 = gr.Textbox(
                            label=i18n("特征检索库文件路径,为空则使用下拉的选择结果"),
                            value="",
                            interactive=True,
                            visible=False
                        )
                        file_index4 = gr.Dropdown(
                            label=i18n("自动检测index路径,下拉式选择(dropdown)"),
                            choices=sorted(index_paths),
                            interactive=True,
                            visible=False
                        )
                        refresh_button.click(
                            fn=lambda: change_choices()[1],
                            inputs=[],
                            outputs=file_index4,
                            api_name="infer_refresh_batch",
                        )
                        # file_big_npy2 = gr.Textbox(
                        #     label=i18n("特征文件路径"),
                        #     value="E:\\codes\\py39\\vits_vc_gpu_train\\logs\\mi-test-1key\\total_fea.npy",
                        #     interactive=True,
                        # )
                        index_rate2 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("检索特征占比"),
                            value=1,
                            interactive=True,
                            visible=False
                        )
                    with gr.Row():
                        resample_sr1 = gr.Slider(
                            minimum=0,
                            maximum=48000,
                            label=i18n("后处理重采样至最终采样率，0为不进行重采样"),
                            value=0,
                            step=1,
                            interactive=True,
                            visible=False
                        )
                        rms_mix_rate1 = gr.Slider(
                            minimum=0,
                            maximum=1,
                            label=i18n("输入源音量包络替换输出音量包络融合比例，越靠近1越使用输出包络"),
                            value=0.21,
                            interactive=True,
                        )
                        protect1 = gr.Slider(
                            minimum=0,
                            maximum=0.5,
                            label=i18n(
                                "保护清辅音和呼吸声，防止电音撕裂等artifact，拉满0.5不开启，调低加大保护力度但可能降低索引效果"
                            ),
                            value=0.33,
                            step=0.01,
                            interactive=True,
                        )
                    with gr.Row():
                        dir_input = gr.Textbox(
                            label=i18n("输入待处理音频文件夹路径(去文件管理器地址栏拷就行了)"),
                            value="./audios",
                        )
                        inputs = gr.File(
                            file_count="multiple", label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹")
                        )
                    with gr.Row():
                        format1 = gr.Radio(
                            label=i18n("导出文件格式"),
                            choices=["wav", "flac", "mp3", "m4a"],
                            value="wav",
                            interactive=True,
                        )
                        but1 = gr.Button(i18n("转换"), variant="primary")
                        vc_output3 = gr.Textbox(label=i18n("输出信息"))
                        but1.click(
                            vc.vc_multi,
                            [
                                spk_item,
                                dir_input,
                                opt_input,
                                inputs,
                                vc_transform1,
                                f0method1,
                                file_index1,
                                file_index2,
                                # file_big_npy2,
                                index_rate1,
                                filter_radius1,
                                resample_sr1,
                                rms_mix_rate1,
                                protect1,
                                format1,
                            ],
                            [vc_output3],
                            api_name="infer_convert_batch",
                        )
            sid0.change(
                fn=vc.get_vc,
                inputs=[sid0, protect0, protect1],
                outputs=[spk_item, protect0, protect1, file_index2, file_index4],
            )
        with gr.TabItem("Download"):
            with gr.TabItem("Download Music"):
                url_input = gr.Textbox(label="URL YT", placeholder="Enter YouTube URL here...")
                with gr.Row():
                    cookie_browser = gr.Dropdown(
                        label="Extract cookies from browser (optional)",
                        choices=["none", "chrome", "firefox", "edge", "brave", "opera", "vivaldi"],
                        value="none",
                        allow_custom_value=False,
                        info="Select a browser to auto-extract login cookies. Requires you to be logged into YouTube in that browser.",
                )
                with gr.Row():
                    cookie_file = gr.File(
                        label="Or upload cookies.txt (optional)",
                        file_types=[".txt"],
                        file_count="single",
                    )
                with gr.Row():
                    optau = gr.Audio(label="Output", type="filepath")
                dl_yt = gr.Button("Download", variant="primary")
                dl_yt.click(
                    fn=yt_download,
                    inputs=[url_input, cookie_file, cookie_browser],
                    outputs=[optau],
                )
            with gr.TabItem("Download Models"):
                with gr.Row():
                    url=gr.Textbox(label="Enter the URL to the Model:")
                with gr.Row():
                    model = gr.Textbox(label="Name your model:")
                    download_button=gr.Button("Download")
                with gr.Row():
                    status_bar=gr.Textbox(label="")
                download_button.click(fn=download_from_url, inputs=[url, model], outputs=[status_bar])
            with gr.Row():
                gr.Markdown(
                """
                ❤️ If you use this and like it, help me keep it.❤️ 
                https://paypal.me/lesantillan
                """
                )
        with gr.TabItem(i18n("训练")):
            with gr.Row():
                with gr.Column():
                    exp_dir1 = gr.Textbox(label=i18n("输入实验名"), value="My-Voice")
                    np7 = gr.Slider(
                        minimum=0,
                        maximum=config.n_cpu,
                        step=1,
                        label=i18n("提取音高和处理数据使用的CPU进程数"),
                        value=int(np.ceil(config.n_cpu / 1.5)),
                        interactive=True,
                    )
                    sr2 = gr.Radio(
                        label=i18n("目标采样率"),
                        choices=["40k", "48k"],
                        value="40k",
                        interactive=True,
                        visible=False
                    )
                    if_f0_3 = gr.Radio(
                        label=i18n("模型是否带音高指导(唱歌一定要, 语音可以不要)"),
                        choices=[True, False],
                        value=True,
                        interactive=True,
                        visible=False
                    )
                    version19 = gr.Radio(
                        label=i18n("版本"),
                        choices=["v1", "v2"],
                        value="v2",
                        interactive=True,
                        visible=False,
                    )
                    trainset_dir4 = gr.Textbox(
                        label=i18n("输入训练文件夹路径"), value='./dataset/'+datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    )
                    easy_uploader = gr.Files(label=i18n("也可批量输入音频文件, 二选一, 优先读文件夹"),file_types=['audio'])
                    but1 = gr.Button(i18n("处理数据"), variant="primary")
                    info1 = gr.Textbox(label=i18n("输出信息"), value="")
                    easy_uploader.upload(fn=upload_to_dataset, inputs=[easy_uploader, trainset_dir4], outputs=[info1, trainset_dir4])
                    gpus6 = gr.Textbox(
                        label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                        value=gpus,
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    gpu_info9 = gr.Textbox(
                        label=i18n("显卡信息"), value=gpu_info, visible=F0GPUVisible
                    )
                    spk_id5 = gr.Slider(
                        minimum=0,
                        maximum=4,
                        step=1,
                        label=i18n("请指定说话人id"),
                        value=0,
                        interactive=True,
                        visible=False
                    )
                    but1.click(
                        preprocess_dataset,
                        [trainset_dir4, exp_dir1, sr2, np7],
                        [info1],
                        api_name="train_preprocess",
                    ) 
                with gr.Column():
                    f0method8 = gr.Radio(
                        label=i18n(
                            "选择音高提取算法:输入歌声可用pm提速,高质量语音但CPU差可用dio提速,harvest质量更好但慢,rmvpe效果最好且微吃CPU/GPU"
                        ),
                        choices=["pm", "harvest", "dio", "rmvpe", "rmvpe_gpu"],
                        value="rmvpe_gpu",
                        interactive=True,
                    )
                    gpus_rmvpe = gr.Textbox(
                        label=i18n(
                            "rmvpe卡号配置：以-分隔输入使用的不同进程卡号,例如0-0-1使用在卡0上跑2个进程并在卡1上跑1个进程"
                        ),
                        value="%s-%s" % (gpus, gpus),
                        interactive=True,
                        visible=F0GPUVisible,
                    )
                    but2 = gr.Button(i18n("特征提取"), variant="primary")
                    info2 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=8)
                    f0method8.change(
                        fn=change_f0_method,
                        inputs=[f0method8],
                        outputs=[gpus_rmvpe],
                    )
                    but2.click(
                        extract_f0_feature,
                        [
                            gpus6,
                            np7,
                            f0method8,
                            if_f0_3,
                            exp_dir1,
                            version19,
                            gpus_rmvpe,
                        ],
                        [info2],
                        api_name="train_extract_f0_feature",
                    )
                with gr.Column():
                    total_epoch11 = gr.Slider(
                        minimum=2,
                        maximum=1000,
                        step=1,
                        label=i18n("总训练轮数total_epoch"),
                        value=150,
                        interactive=True,
                    )
                    gpus16 = gr.Textbox(
                            label=i18n("以-分隔输入使用的卡号, 例如   0-1-2   使用卡0和卡1和卡2"),
                            value="0",
                            interactive=True,
                            visible=True
                        )
                    but3 = gr.Button(i18n("训练模型"), variant="primary")
                    but4 = gr.Button(i18n("训练特征索引"), variant="primary")
                    info3 = gr.Textbox(label=i18n("输出信息"), value="", max_lines=10)
                    with gr.Accordion(label=i18n("常规设置"), open=False):
                        save_epoch10 = gr.Slider(
                            minimum=1,
                            maximum=50,
                            step=1,
                            label=i18n("保存频率save_every_epoch"),
                            value=25,
                            interactive=True,
                        )
                        batch_size12 = gr.Slider(
                            minimum=1,
                            maximum=40,
                            step=1,
                            label=i18n("每张显卡的batch_size"),
                            value=default_batch_size,
                            interactive=True,
                        )
                        if_save_latest13 = gr.Radio(
                            label=i18n("是否仅保存最新的ckpt文件以节省硬盘空间"),
                            choices=[i18n("是"), i18n("否")],
                            value=i18n("是"),
                            interactive=True,
                            visible=False
                        )
                        if_cache_gpu17 = gr.Radio(
                            label=i18n(
                                "是否缓存所有训练集至显存. 10min以下小数据可缓存以加速训练, 大数据缓存会炸显存也加不了多少速"
                            ),
                            choices=[i18n("是"), i18n("否")],
                            value=i18n("否"),
                            interactive=True,
                        )
                        if_save_every_weights18 = gr.Radio(
                            label=i18n("是否在每次保存时间点将最终小模型保存至weights文件夹"),
                            choices=[i18n("是"), i18n("否")],
                            value=i18n("是"),
                            interactive=True,
                        )
                    with gr.Row():
                        download_model = gr.Button('5.Download Model')
                    with gr.Row():
                        model_files = gr.Files(label='Your Model and Index file can be downloaded here:')
                        download_model.click(fn=download_model_files, inputs=[exp_dir1], outputs=[model_files, info3])
                    with gr.Row():
                        pretrained_G14 = gr.Textbox(
                            label=i18n("加载预训练底模G路径"),
                            value="assets/pretrained_v2/f0G40k.pth",
                            interactive=True,
                            visible=False
                        )
                        pretrained_D15 = gr.Textbox(
                            label=i18n("加载预训练底模D路径"),
                            value="assets/pretrained_v2/f0D40k.pth",
                            interactive=True,
                            visible=False
                        )
                        sr2.change(
                            change_sr2,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15],
                        )
                        version19.change(
                            change_version19,
                            [sr2, if_f0_3, version19],
                            [pretrained_G14, pretrained_D15, sr2],
                        )
                        if_f0_3.change(
                            change_f0,
                            [if_f0_3, sr2, version19],
                            [f0method8, pretrained_G14, pretrained_D15],
                        )
                    with gr.Row():
                        but5 = gr.Button(i18n("一键训练"), variant="primary", visible=False)
                        but3.click(
                            click_train,
                            [
                                exp_dir1,
                                sr2,
                                if_f0_3,
                                spk_id5,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                            ],
                            info3,
                            api_name="train_start",
                        )
                        but4.click(train_index, [exp_dir1, version19], info3)
                        but5.click(
                            train1key,
                            [
                                exp_dir1,
                                sr2,
                                if_f0_3,
                                trainset_dir4,
                                spk_id5,
                                np7,
                                f0method8,
                                save_epoch10,
                                total_epoch11,
                                batch_size12,
                                if_save_latest13,
                                pretrained_G14,
                                pretrained_D15,
                                gpus16,
                                if_cache_gpu17,
                                if_save_every_weights18,
                                version19,
                                gpus_rmvpe,
                            ],
                            info3,
                            api_name="train_start_all",
                        )

    if config.iscolab:
        app.queue().launch(share=True)
    else:
        app.queue(concurrency_count=511, max_size=1022).launch(
            server_name="0.0.0.0",
            inbrowser=not config.noautoopen,
            server_port=config.listen_port,
            quiet=True,
        )
