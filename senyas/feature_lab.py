"""Camera-based inspection and annotation for the development feature schema."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from .camera import live_camera,decode_camera_attempt
from .config import PROJECT_ROOT
from .handshape_samples import HAND_SHAPES,save_labelled_sample
from .sequences import SequenceExtractor,SEQUENCE_SPECS,circle_measurements

CAPTURE_FRAMES=121


def main():
    st.set_page_config(page_title='Senyas · Feature lab',page_icon='🤟',layout='wide',initial_sidebar_state='expanded')
    st.markdown('<style>'+Path(__file__).with_name('theme.css').read_text(encoding='utf-8')+'</style>',unsafe_allow_html=True)
    with st.sidebar:
        st.title('Senyas')
        st.caption('Development feature lab')
        st.markdown('[Open learning app](http://127.0.0.1:8501/)')
        st.write('Inspect geometry and collect reusable hand-shape examples.')
        st.caption('This lab does not recognize a word or train a classifier.')
    st.title('Engineered feature lab')
    st.write('Record a hand shape, inspect the measured features, then optionally save a labelled clip.')
    st.caption('Camera target: 30 FPS. Actual capture rate and processing throughput are measured separately. Palm-facing labels, detailed finger states, and hand-shape names require reviewed examples.')
    settings=st.columns(2)
    size=settings[0].select_slider('Analysis size',options=[360,480,640],value=480)
    complexity=settings[1].selectbox('Pose model',options=[1,0],format_func=lambda value:'Lite (compare tracking)' if value==0 else 'Full (benchmark default)')
    context=f'semantic-lab:{size}:{complexity}'
    payload=live_camera(context,st.session_state.get('lab_ack'),frame_count=CAPTURE_FRAMES,key='semantic_lab_camera')
    if isinstance(payload,dict) and payload.get('context')==context and payload.get('capture_id')!=st.session_state.get('lab_ack'):
        st.session_state.lab_ack=payload.get('capture_id')
        try:
            attempt=decode_camera_attempt(payload,context,expected_frame_count=(16,CAPTURE_FRAMES))
            with st.spinner('Measuring the ordered feature sequence…'):
                with SequenceExtractor(target_fps=30,max_image_side=size,model_complexity=complexity) as extractor:
                    sequence=extractor.extract_frames(attempt.frames,attempt.timestamps)
            st.session_state.lab_result={'sequence':sequence,'frames':attempt.frames,'capture_id':attempt.capture_id,'context':context}
            st.session_state.pop('lab_error',None)
        except Exception as exc:
            st.session_state.lab_error=str(exc)
            st.session_state.pop('lab_result',None)
        st.rerun()
    if st.session_state.get('lab_error'):
        st.error(st.session_state.lab_error)
    result=st.session_state.get('lab_result')
    if not result or result['context']!=context:
        return
    sequence=result['sequence'];metadata=sequence.metadata
    columns=st.columns(3)
    columns[0].metric('Capture sampling',f'{metadata["sampled_fps"]:.1f} FPS')
    columns[1].metric('Processing throughput',f'{metadata["throughput_fps"]:.1f} FPS')
    columns[2].metric('Frames with a hand',f'{metadata["hand_coverage"]:.0%}')
    if metadata['sampled_fps']<24 or metadata['throughput_fps']<24:
        st.info('This attempt fell below the accepted 24 FPS target for capture or processing. Try the smaller analysis size and keep this tab active.')
    if metadata['hand_coverage']<.5:
        st.warning('Hand tracking was missing in more than half the frames. Review the interval before saving a label.')
    groups=sorted({spec.group for spec in SEQUENCE_SPECS})
    tabs=st.tabs(['Measurements','Movement','Save labelled sample'])
    with tabs[0]:
        group=st.selectbox('Feature family',groups)
        frame_index=st.slider('Inspect frame',0,len(sequence.timestamps)-1,len(sequence.timestamps)-1)
        st.caption(f'Frame timestamp: {sequence.timestamps[frame_index]:.3f} seconds. Missing values mean unobserved; proximity is not confirmed contact.')
        rows=[{'feature':spec.name,'value':float(sequence.values[frame_index,i]) if np.isfinite(sequence.values[frame_index,i]) else None,
               'unit':spec.unit,'meaning':spec.meaning,'experimental':spec.experimental}
              for i,spec in enumerate(SEQUENCE_SPECS) if spec.group==group]
        st.dataframe(pd.DataFrame(rows),hide_index=True,use_container_width=True)
    with tabs[1]:
        motion=[f'{side}.wrist.body_{axis}' for side in ['left','right'] for axis in 'xy']
        st.line_chart(pd.DataFrame(sequence.values[:,[sequence.feature_names.index(name) for name in motion]],index=sequence.timestamps,columns=motion))
        st.caption('Positions are shoulder-normalized. Positive x is unmirrored image-right; positive y is up. Gaps remain missing.')
        for side in ['left','right']:
            st.write(side.capitalize()+' wrist trajectory')
            st.json(circle_measurements(sequence,side))
        st.caption('A turning trajectory alone does not establish a complete circle. Finger motion is measured relative to the moving palm; nodding and shaking labels remain untrained.')
    with tabs[2]:
        st.write('Hold one shape during the labelled interval. Choose labels independently for each visible hand; leave the other hand unlabelled.')
        label_columns=st.columns(2)
        labels={}
        for side,column in zip(['left','right'],label_columns):
            label=column.selectbox(side.capitalize()+' hand shape',['Unlabelled',*HAND_SHAPES],key='label_'+side)
            if label!='Unlabelled':labels[side]=label
        interval=st.slider('Labelled interval in seconds',0.,float(sequence.timestamps[-1]),(0.,float(sequence.timestamps[-1])),step=.05)
        signer=st.text_input('Signer or recording-session ID',value='local-session')
        st.caption('You can label several intervals in the same clip. Start with three to five clips, include repeated examples of each shape, and add more signers later. Every interval from one recording must remain in the same data split.')
        if st.button('Save labelled clip',type='primary',disabled=not labels):
            try:
                paths=save_labelled_sample(PROJECT_ROOT/'data/handshape_samples',result['capture_id'],sequence,result['frames'],labels,interval,signer)
                st.success('Saved a silent clip, timestamped landmarks/features, and its hand-shape labels.')
                st.code(str(paths['json']))
            except Exception as exc:
                st.error(str(exc))
