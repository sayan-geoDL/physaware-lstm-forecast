#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extra utility functions for the weather forecasting pipeline.

Provides:
- GPU/CPU temperature monitoring.
- Parsing helpers for ranges/lists.
- Pareto efficiency filtering and natural sorting.
- File purging by prefix.
- PDF report generation (CV runs, single model, ensemble).

Notes:
- Requires `nvidia-smi` for GPU temperature.
- Uses `reportlab` for PDF generation.
- Includes a legacy `grid()` function (kept for compatibility).
"""
import numpy as np
import subprocess
import psutil
import ast
import os
import glob
import re
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,Table,TableStyle,
    Paragraph,Spacer,Image,PageBreak,LongTable
)
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib.utils import ImageReader
import logging
logger=logging.getLogger(__name__)
####################### getting temperature of the system #####################
def get_gpu_temperature():
    """
    Returns the current GPU temperature in degrees Celsius using `nvidia-smi`.

    Returns:
        int or None: GPU temperature if available, else None.

    Notes:
        - Requires NVIDIA GPU and `nvidia-smi` in system PATH.
        - Only the first GPU is queried.
        - Logs errors to the logger if fetching fails.
        - Does not save any files.
    """
    try:
        output=subprocess.check_output(['nvidia-smi','--query-gpu=temperature.gpu','--format=csv,noheader,nounits'])
        temp_str=output.decode('utf-8').strip()
        temp=int(temp_str.split('\n')[0])
        return temp
    except Exception as e:
        logger.info(f"Error fetching GPU temperature: {e}")
        return None
def get_cpu_temperature():
    """
    Returns the current CPU temperature in degrees Celsius using `psutil`.

    Returns:
        float or None: CPU temperature if available, else None.

    Notes:
        - Uses `psutil.sensors_temperatures()`.
        - Searches for 'coretemp' or 'Package id 0'.
        - Logs errors to the logger if fetching fails.
        - Does not save any files.
    """
    temps=psutil.sensors_temperatures()
    if not temps:
        return None
    for name,entries in temps.items():
        for entry in entries:
            if entry.label=='Package id 0' or 'coretemp' in name.lower():
                return entry.current
    return None
##################### Parsing strings to tuples or accepting lists ############
def parse_range_or_list(entry):
    """
    Parses a string of format "(start, stop, step)" or accepts a list directly to generate a list of values.

    Args:
        entry (str or list): List of values or a string representing a 3-element tuple.

    Returns:
        list: List of values generated using np.arange or range.

    Raises:
        ValueError: If parsing fails or format is invalid.

    Notes:
        - Does not save any files.
    """
    if isinstance(entry,list):
        return entry
    elif isinstance(entry,str):
        try:
            parsed=ast.literal_eval(entry)
            if not isinstance(parsed,tuple) or len(parsed) != 3:
                raise ValueError("Expected a 3-element tuple string,e.g.,'(0.0,0.01,0.002)'")
            start,stop,step=parsed
            if any(isinstance(x,float) for x in parsed):
                return list(np.arange(start,stop,step))
            else:
                return list(range(start,stop,step))
        except Exception as e:
            raise ValueError(f"Failed to parse range from string '{entry}': {e}")
    else:
        raise ValueError("Only list or '(start,stop,step)' string formats are supported.")


##################### Generating dict for grid search #########################
def grid(hidden_num,layer_num,learn_rate,weight_decay,lambda_phys):
    """
    Prepare a hyperparameter grid for LSTM training,accepting lists or string-tuple ranges.

    Parameters
    ----------
    hidden_num : list or str
        Hidden layer sizes,or a string like '(32,128,32)'.
    layer_num : list or str
        Number of layers,or a string like '(1,4,1)'.
    learn_rate : list or str
        Learning rates,or a string like '(0.0001,0.001,0.0002)'.
    weight_decay : list or str
        Weight decay values,or a string like '(0.0,0.001,0.0002)'.

    Returns
    -------
    grid_params : dict
        Dictionary containing lists of hyperparameter values:
        {
            'hidden_size': [...],
            'num_layers': [...],
            'learning_rate': [...],
            'weight_decay': [...]
        }
    """
    grid_params={
        'hidden_size': parse_range_or_list(hidden_num),
        'num_layers': parse_range_or_list(layer_num),
        'learning_rate': parse_range_or_list(learn_rate),
        'weight_decay': parse_range_or_list(weight_decay),
        'lambda_physics': parse_range_or_list(lambda_phys),
    }
    return grid_params
############################### pareto efficiency ##############################
def is_pareto_efficient(costs):
    """
    Determines Pareto-efficient points in a cost array.

    Args:
        costs (np.ndarray): Array of shape (n_points, n_costs).

    Returns:
        np.ndarray: Boolean mask indicating Pareto-efficient points.

    Notes:
        - Does not save any files.
    """
    is_efficient=np.ones(costs.shape[0],dtype=bool)
    for i,c in enumerate(costs):
        if is_efficient[i]:
            dominates=np.all(costs[is_efficient] <= c,axis=1) & np.any(costs[is_efficient] < c,axis=1)
            is_efficient[is_efficient]=~dominates
            is_efficient[i]=True  # Keep current point
    return is_efficient
def natural_key(string):
    """
    Helper for natural sorting of file names like fold_1, fold_2, etc.

    Args:
        string (str): Input string.

    Returns:
        list: List of string and integer parts for sorting.

    Notes:
        - Does not save any files.
    """
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)',string)]
########################### file purger
def delete_files_with_prefix(directory: str,prefix: str,logger=None):
    """
    Deletes all files in a directory that start with the given prefix.

    Args:
        directory (str): Path to the directory.
        prefix (str): Filename prefix to match.
        logger (logging.Logger, optional): Logger for info messages.

    Returns:
        list: List of deleted file paths.

    Saves:
        - Deletes matching files from the specified directory.
        - Logs actions to the provided logger if given.
    """
    pattern=os.path.join(directory,prefix+"*")
    deleted_files=[]

    for file in glob.glob(pattern):
        if os.path.isfile(file):
            try:
                os.remove(file)
                deleted_files.append(file)
            except Exception as e:
                logger.info(f"Error deleting {file}: {e}")
    if logger:
        if deleted_files:
            logger.info(f"Deleted {len(deleted_files)} files with prefix '{prefix}' from {directory}")
        else:
            logger.info(f"No files found with prefix '{prefix}' in {directory}")

    return deleted_files


########### cv report
def generate_hyperparam_report(csv_file,plot_dir,output_pdf="hyperparam_report.pdf"):
    """
    Generates a PDF report summarizing hyperparameter search results and plots.

    Args:
        csv_file (str): Path to cross-validation results CSV.
        plot_dir (str): Directory containing fold and blend plots.
        output_pdf (str): Output PDF file path.

    Saves:
        - PDF report to the specified output_pdf (default: ./hyperparam_report.pdf).
        - Includes summary tables, search space, scatter plots, and fold plots.
    """
    df=pd.read_csv(csv_file)
    df=df.round(6)
    doc=SimpleDocTemplate(output_pdf,pagesize=landscape(A4),rightMargin=20,leftMargin=20)
    styles=getSampleStyleSheet()
    elements=[]
    elements.append(Paragraph("Summary of Hyperparameter Runs",styles["Heading1"]))
    data=[df.columns.tolist()]+df.astype(str).values.tolist()
    page_width=landscape(A4)[0]-40  # account for margins
    col_widths=[page_width/len(df.columns)]*len(df.columns)
    highlight_cols=[]
    for col_name in ["mean_rmse","mean_overfit_score"]:
        if col_name in df.columns:
            highlight_cols.append(df.columns.get_loc(col_name))
    table=Table(data,colWidths=col_widths,repeatRows=1)
    style=[
    ('BACKGROUND',(0,0),(-1,0),colors.lightblue),# header background
    ('TEXTCOLOR',(0,0),(-1,0),colors.black),
    ('ALIGN',(0,0),(-1,-1),'CENTER'),
    ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
    ('FONTSIZE',(0,0),(-1,-1),8),
    ('GRID',(0,0),(-1,-1),0.25,colors.grey),]
    for col in highlight_cols:
        style.append(('BACKGROUND',(col,1),(col,-1),colors.lightyellow))  # body highlight
        style.append(('BACKGROUND',(col,0),(col,0),colors.yellow))
    table.setStyle(TableStyle(style))
    elements.append(table)
    elements.append(PageBreak())

    search_space={
        "hidden_size": {"type": "int","low": 1,"high": 100,"step": 1},
        "num_layers": {"type": "int","low": 1,"high": 3,"step": 1},
        "learning_rate": {"type": "float","low": 1e-4,"high": 5e-2,"log": True},
        "weight_decay": {"type": "float","low": 0.0001,"high": 1,"log": True},
        "lambda_physics": {"type": "float","low": 0.001,"high": 10.0,"log": True},
    }
    elements.append(Paragraph("Search Space",styles["Heading1"]))
    for k,v in search_space.items():
        elements.append(Paragraph(f"<b>{k}</b>: {v}",styles["Normal"]))
    image_path="./plots/cv/scatters.png"
    if os.path.exists(image_path):
        elements.append(Spacer(1,0.3*inch))
        elements.append(Image(image_path,width=7*inch,height=4.5*inch))
    elements.append(PageBreak())

    for rank,row in df.iterrows():
        rank_num=rank+1  # start from 1
        elements.append(Paragraph(f"Rank {rank_num} Model",styles["Heading1"]))

        param_data=[[col,str(row[col])] for col in df.columns]
        param_table=Table(param_data,colWidths=[2*inch,page_width-2*inch])
        param_table.setStyle(TableStyle([
            ('GRID',(0,0),(-1,-1),0.25,colors.grey),
            ('FONTNAME',(0,0),(-1,-1),'Helvetica'),
            ('FONTSIZE',(0,0),(-1,-1),9),
            ('ALIGN',(0,0),(-1,-1),'LEFT'),
        ]))
        elements.append(param_table)
        elements.append(Spacer(1,0.2*inch))

        rank_plots=[f for f in os.listdir(plot_dir) if f.startswith(f"rank{rank_num}_fold")]
        rank_plots=sorted(rank_plots,key=natural_key)

        blend_plot=f"rank{rank_num}blend_graph.png"
        if blend_plot in os.listdir(plot_dir):
            rank_plots.append(blend_plot)

        for plot in rank_plots:
            elements.append(Image(os.path.join(plot_dir,plot),width=6*inch,height=3*inch))
            elements.append(Spacer(1,0.2*inch))

        elements.append(PageBreak())
    doc.build(elements)
    logger.info(f"cv report saved as:,{os.path.abspath(output_pdf)}")
############################### test report
def generate_single_model_report():
    """
    Generates a PDF report summarizing single model performance.

    Loads:
        - Metrics from ./out/train_test/metrics.csv
        - Loss plot from ./plots/train/train_val_loss.png
        - Distribution plot from ./plots/train/distributions.png
        - Time series plot from ./plots/train/time_series.png

    Saves:
        - PDF report to ./out/train_test/single_model_report.pdf
        - Includes metrics table, loss curve, distributions, and time series plots.
    """
    PAGE_WIDTH,PAGE_HEIGHT=A4
    margin=40
    figure_counter={"n": 1}

    def get_resized_image(path,max_width,max_height):
        """Load image and scale proportionally to fit."""
        img_reader=ImageReader(path)
        iw,ih=img_reader.getSize()
        scale=min(max_width/iw,max_height/ih,1.0)
        return Image(path,width=iw*scale,height=ih*scale)

    def add_figure(story,img_path,caption,max_width,max_height):
        """Append image+caption with auto-numbering."""
        img=get_resized_image(img_path,max_width,max_height)
        story.append(img)
        story.append(Spacer(1,6))
        styles=getSampleStyleSheet()
        caption_style=ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=9,
            alignment=1,# center
        )
        story.append(Paragraph(
            f"Figure {figure_counter['n']}: {caption}",
            caption_style
        ))
        story.append(Spacer(1,12))
        figure_counter["n"]+=1

    stats_csv="./out/train_test/metrics.csv"
    loss_img="./plots/train/train_val_loss.png"
    dist_img="./plots/train/distributions.png"
    ts_img="./plots/train/time_series.png"
    
    if not os.path.exists(stats_csv):
        logger.info(f"[generate_single_model_report] missing CSV: {stats_csv}")
        return
    if not os.path.exists(loss_img):
        logger.info(f"[generate_single_model_report] missing image: {loss_img}")
        return
    if not os.path.exists(dist_img):
        logger.info(f"[generate_single_model_report] missing image: {dist_img}")
        return
    if not os.path.exists(ts_img):
        logger.info(f"[generate_single_model_report] missing image: {ts_img}")
        return

    df=pd.read_csv(stats_csv)
    data=[df.columns.to_list()]+df.values.tolist()

    pdf=SimpleDocTemplate(
        "./out/train_test/single_model_report.pdf",
        pagesize=A4
    )
    styles=getSampleStyleSheet()
    story=[]

    story.append(Paragraph("<b>Single Model Report</b>",styles['Title']))
    story.append(Spacer(1,20))

    table=Table(data)
    table.setStyle(TableStyle([
        ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
        ("GRID",(0,0),(-1,-1),0.5,colors.black),
        ("ALIGN",(0,0),(-1,-1),"CENTER"),
    ]))
    story.append(table)

    story.append(PageBreak())
    add_figure(
        story,
        loss_img,
        "Training and Validation Loss",
        PAGE_WIDTH-2*margin,
        PAGE_HEIGHT-2*margin
    )

    story.append(PageBreak())
    max_width=(PAGE_WIDTH-3*margin)/2
    max_height=PAGE_HEIGHT-2*margin

    img2=get_resized_image(dist_img,max_width,max_height*0.85)
    img3=get_resized_image(ts_img,max_width,max_height*0.85)

    side_by_side=Table([[img2,img3]],
                         colWidths=[max_width,max_width])
    side_by_side.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
    story.append(side_by_side)
    story.append(Spacer(1,6))

    caption_style=ParagraphStyle(
        "Caption",
        parent=styles["Normal"],
        fontSize=9,
        alignment=1,# center
    )
    story.append(Paragraph(
        f"Figure {figure_counter['n']}: Distributions (left) and Time Series (right)",
        caption_style
    ))
    figure_counter["n"]+=1

    # build
    pdf.build(story)
    logger.info("Single model report generated!")
###################### ensemble report
def generate_ensemble_report(var_names,ranks,
                             metrics_csv="./out/train_test/ensemble_metrics.csv",
                             pareto_csv="./out/train_test/cv_result.csv",
                             output_pdf="./out/train_test/ensemble_report.pdf",
                             plot_dir="./plots/train"):
    """
     Generates a PDF report summarizing ensemble model performance.

     Args:
         var_names (list): List of variable names (e.g., ['dwpt', 'temp', ...]).
         ranks (list): List of model ranks to include.
         metrics_csv (str): Path to ensemble metrics CSV.
         pareto_csv (str): Path to Pareto-optimal hyperparameters CSV.
         output_pdf (str): Output PDF file path.
         plot_dir (str): Directory containing plots.

     Loads:
         - Metrics and hyperparameters from CSVs.
         - Plots from plot_dir (distributions, time series, loss curves).

     Saves:
         - PDF report to output_pdf (default: ./out/train_test/ensemble_report.pdf)
         - Includes metrics tables, ensemble and per-rank plots, and parameter tables.
    """
    if not var_names or len(var_names) == 0:
        raise ValueError("var_names cannot be None or empty")
    
    if ranks == 'all':
        if not os.path.exists(pareto_csv):
            raise FileNotFoundError(f"Pareto CSV file not found: {pareto_csv}")
        df_temp = pd.read_csv(pareto_csv)
        if len(df_temp) == 0:
            raise ValueError(f"Pareto CSV file is empty: {pareto_csv}")
        ranks = list(range(1, len(df_temp) + 1))
    
    if ranks is None or len(ranks) == 0:
        raise ValueError("ranks cannot be None or empty. Provide a list of ranks or use 'all'.")
    PAGE_WIDTH,PAGE_HEIGHT=A4
    margin=36
    caption_space=60

    def get_resized_image(path,max_width,max_height):
        img_reader=ImageReader(path)
        iw,ih=img_reader.getSize()
        allowed_height=max(1,max_height-caption_space)
        scale=min(max_width/iw,allowed_height/ih,1.0)*0.95
        new_w,new_h=iw*scale,ih*scale
        return Image(path,width=new_w,height=new_h)

    styles=getSampleStyleSheet()
    figure_counter={"n": 1}

    def add_figure(story,img_path,caption,max_width,max_height):
        if not os.path.exists(img_path):
            logger.info(f"[generate_ensemble_report] missing image: {img_path}")
            return
        img=get_resized_image(img_path,max_width,max_height)
        story.append(img)
        story.append(Spacer(1,6))
        caption_style=ParagraphStyle(
            "Caption",
            parent=styles["Normal"],
            fontSize=9,
            alignment=1,# center
            leading=11
        )
        story.append(Paragraph(f"Figure {figure_counter['n']}: {caption}",caption_style))
        story.append(Spacer(1,12))
        figure_counter["n"]+=1

    df_metrics=pd.read_csv(metrics_csv)
    df_pareto=pd.read_csv(pareto_csv)

    pdf=SimpleDocTemplate(output_pdf,
                             pagesize=A4,
                             leftMargin=margin,
                             rightMargin=margin,
                             topMargin=margin,
                             bottomMargin=margin)

    story=[]

    story.append(Paragraph("<b>Ensemble Model Report</b>",styles['Title']))
    story.append(Spacer(1,12))

    usable_width=PAGE_WIDTH-2*margin

    for var in var_names:
        story.append(PageBreak())
        story.append(Paragraph(f"<b>Metrics for {var}</b>",styles['Heading2']))
        story.append(Spacer(1,6))
     
        cols=['rank',f'{var}_train_rmse',f'{var}_test_rmse',f'{var}_train_r2',f'{var}_test_r2']
        table_data=[cols]+df_metrics[cols].astype(str).values.tolist()
        table=LongTable(table_data,colWidths=[usable_width*0.2]*5,repeatRows=1)
        table.setStyle(TableStyle([
            ("BACKGROUND",(0,0),(-1,0),colors.lightgrey),
            ("ALIGN",(0,0),(-1,-1),"CENTER"),
            ("GRID",(0,0),(-1,-1),0.25,colors.black),
            ("VALIGN",(0,0),(-1,-1),"MIDDLE"),
            ("FONTSIZE",(0,0),(-1,-1),9)
            ]))
        story.append(table)

    story.append(PageBreak())
    add_figure(story,
                os.path.join(plot_dir,"ensemble_distributions.png"),
                "Distribution of Observed vs Predicted (Ensemble)",
                usable_width,
                PAGE_HEIGHT-2*margin)

    for var in var_names:
        story.append(PageBreak())
        train_img=os.path.join(plot_dir,f"ensemble_{var}_train_ts.png")
        add_figure(story,train_img,f"Ensemble Train Time Series for {var}",usable_width,(PAGE_HEIGHT-2*margin)/2)
        add_figure(story,os.path.join(plot_dir,f"ensemble_{var}_test_ts.png"),
                   f"Ensemble Test Time Series for {var}",usable_width,(PAGE_HEIGHT-2*margin)/2)

    for rank in ranks:
        idx=rank-1
        story.append(PageBreak())
        story.append(Paragraph(f"<b>Rank {rank}</b>",styles['Heading1']))
        story.append(Spacer(1,8))

        params_row=df_pareto.iloc[idx].to_dict() if idx < len(df_pareto) else {}
        metrics_row=df_metrics.iloc[idx].to_dict() if idx < len(df_metrics) else {}
        combined={**params_row,**metrics_row}
        table_data=[[str(k),str(v)] for k,v in combined.items()] if combined else [["note","no data for this rank"]]
        small_col_widths=[usable_width*0.35,usable_width*0.65]
        small_table=Table(table_data,colWidths=small_col_widths)
        small_table.setStyle(TableStyle([
            ("GRID",(0,0),(-1,-1),0.25,colors.black),
            ("ALIGN",(0,0),(-1,-1),"LEFT"),
            ("FONTSIZE",(0,0),(-1,-1),9),
        ]))
        story.append(small_table)

        story.append(PageBreak())
        add_figure(story,
                   os.path.join(plot_dir,f"rank{rank}_train_val_loss.png"),
                   f"Training/Hold Out Loss for Rank {rank}",
                   usable_width,
                   PAGE_HEIGHT-2*margin)

        story.append(PageBreak())
        dist_img=os.path.join(plot_dir,f"rank{rank}_distributions.png")
        ts_img=os.path.join(plot_dir,f"rank{rank}_time_series.png")
        if os.path.exists(dist_img) and os.path.exists(ts_img):
            gutter=12
            half_width=(usable_width-gutter)/2
            img1=get_resized_image(dist_img,half_width,PAGE_HEIGHT-2*margin)
            img2=get_resized_image(ts_img,half_width,PAGE_HEIGHT-2*margin)
            side_by_side=Table([[img1,img2]],colWidths=[half_width,half_width])
            side_by_side.setStyle(TableStyle([("VALIGN",(0,0),(-1,-1),"MIDDLE")]))
            story.append(side_by_side)
            story.append(Spacer(1,6))
            cap_style=ParagraphStyle("cap",parent=styles["Normal"],fontSize=9,alignment=1)
            story.append(Paragraph(f"Figure {figure_counter['n']}: Rank {rank} Distributions (left) and Time Series (right)",cap_style))
            figure_counter["n"]+=1
        else:
            add_figure(story,dist_img,f"Rank {rank} Distributions",usable_width,PAGE_HEIGHT-2*margin)
            add_figure(story,ts_img,f"Rank {rank} Time Series",usable_width,PAGE_HEIGHT-2*margin)

    pdf.build(story)
    logger.info(f"[generate_ensemble_report] saved: {os.path.abspath(output_pdf)}")
