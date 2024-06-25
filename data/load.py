import polars as pl
import zipfile


def load_parquets_from_zip(path: str):
    """
    Load all the parquet file in a zip file
    
    Parameters
    ----------
    path : str
        path to the zip file containing the parquets.

    Returns
    dict
        a dictionary with the name of the parquet file less the .parquet and a polars dataframe.
    """
    res = {}
    with zipfile.ZipFile(path, 'r') as z:
        for f_name in z.namelist():
            if f_name.startswith('__') or not f_name.endswith('.parquet'):
                continue
            #name = f_name.split('/')[-1][:-8]
            name = f_name[:-8]
            with z.open(f_name) as f:
                res[name] = pl.read_parquet(f) 
    return res


def merge_article_embs(*dfs):
    res = None
    for ds in dfs:
        if res is None:
            res = ds
            continue
        res = res.join(ds, on='article_id')
    columns = [x for x in res.columns if x != 'article_id']
    res = res.with_columns(embeddings=pl.concat_list(columns))
    res = res.select('article_id', 'embeddings')
    return res


def merge_article_with_imgs(text, imgs, col='image_embedding'):
    if col != 'image_embedding':
        imgs = imgs.rename({col: 'image_embedding'})
    embsize = len(imgs['image_embedding'][0])
    res = text.join(imgs, on='article_id', how='outer')
    res = res.with_columns(has_image=pl.col('image_embedding').\
                           is_not_null().\
                            map_elements(lambda x: 1 if x else 0, 
                                         return_dtype=pl.Int64))
    res = res.with_columns(pl.col('image_embedding').fill_null([0.0] * embsize))
    columns = [x for x in res.columns if not x.startswith('article_id') and not x == 'has_image']
    res = res.with_columns(embeddings=pl.concat_list(columns))
    res = res.select('article_id', 'embeddings', 'has_image')
    return res