'''
Until the definition of ``Client``, direct import from
https://github.com/dstndstn/astrometry.net/blob/master/net/client/client.py
(version: 2018-07-05 commit #26668f3)
with dropping support for python < 3.6.

Some tweak made in the definition of ``query_nova``.
'''
import os
import time
import base64

from urllib.parse import urlencode, quote
from urllib.request import urlopen, Request
from urllib.error import HTTPError

# from exceptions import Exception
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication

from email.encoders import encode_noop

import json

__all__ = ["query_nova"]


def json2python(data):
    try:
        return json.loads(data)
    except:
        pass
    return None


python2json = json.dumps


class MalformedResponse(Exception):
    pass


class RequestError(Exception):
    pass


class Client(object):
    default_url = 'http://nova.astrometry.net/api/'

    def __init__(self, apiurl=default_url):
        self.session = None
        self.apiurl = apiurl

    def get_url(self, service):
        return self.apiurl + service

    def send_request(self, service, args={}, file_args=None):
        '''
        service: str
        args: dict
        '''
        if self.session is not None:
            args.update({'session': self.session})
        print('Python:', args)
        json = python2json(args)
        print('Sending json:', json)
        url = self.get_url(service)
        print('Sending to URL:', url)

        # If we're sending a file, format a multipart/form-data
        if file_args is not None:
            # Make a custom generator to format it the way we need.
            from io import BytesIO
            from email.generator import BytesGenerator as TheGenerator

            m1 = MIMEBase('text', 'plain')
            m1.add_header('Content-disposition',
                          'form-data; name="request-json"')
            m1.set_payload(json)
            m2 = MIMEApplication(file_args[1], 'octet-stream', encode_noop)
            m2.add_header('Content-disposition',
                          'form-data; name="file"; filename="%s"' % file_args[0])
            mp = MIMEMultipart('form-data', None, [m1, m2])

            class MyGenerator(TheGenerator):
                def __init__(self, fp, root=True):
                    # don't try to use super() here; in py2 Generator is not a
                    # new-style class.  Yuck.
                    TheGenerator.__init__(self, fp, mangle_from_=False,
                                          maxheaderlen=0)
                    self.root = root

                def _write_headers(self, msg):
                    # We don't want to write the top-level headers;
                    # they go into Request(headers) instead.
                    if self.root:
                        return
                    # We need to use \r\n line-terminator, but Generator
                    # doesn't provide the flexibility to override, so we
                    # have to copy-n-paste-n-modify.
                    for h, v in msg.items():
                        self._fp.write(('%s: %s\r\n' % (h, v)).encode())
                    # A blank line always separates headers from body
                    self._fp.write('\r\n'.encode())

                # The _write_multipart method calls "clone" for the
                # subparts.  We hijack that, setting root=False
                def clone(self, fp):
                    return MyGenerator(fp, root=False)

            fp = BytesIO()
            g = MyGenerator(fp)
            g.flatten(mp)
            data = fp.getvalue()
            headers = {'Content-type': mp.get('Content-type')}

        else:
            # Else send x-www-form-encoded
            data = {'request-json': json}
            print('Sending form data:', data)
            data = urlencode(data)
            data = data.encode('utf-8')
            print('Sending data:', data)
            headers = {}

        request = Request(url=url, headers=headers, data=data)

        try:
            f = urlopen(request)
            txt = f.read()
            print('Got json:', txt)
            result = json2python(txt)
            print('Got result:', result)
            stat = result.get('status')
            print('Got status:', stat)
            if stat == 'error':
                errstr = result.get('errormessage', '(none)')
                raise RequestError('server error message: ' + errstr)
            return result
        except HTTPError as e:
            print('HTTPError', e)
            txt = e.read()
            open('err.html', 'wb').write(txt)
            print('Wrote error text to err.html')

    def login(self, apikey):
        args = {'apikey': apikey}
        result = self.send_request('login', args)
        sess = result.get('session')
        print('Got session:', sess)
        if not sess:
            raise RequestError('no session in result')
        self.session = sess

    def _get_upload_args(self, **kwargs):
        args = {}
        for key, default, typ in [('allow_commercial_use', 'd', str),
                                  ('allow_modifications', 'd', str),
                                  ('publicly_visible', 'y', str),
                                  ('scale_units', None, str),
                                  ('scale_type', None, str),
                                  ('scale_lower', None, float),
                                  ('scale_upper', None, float),
                                  ('scale_est', None, float),
                                  ('scale_err', None, float),
                                  ('center_ra', None, float),
                                  ('center_dec', None, float),
                                  ('parity', None, int),
                                  ('radius', None, float),
                                  ('downsample_factor', None, int),
                                  ('tweak_order', None, int),
                                  ('crpix_center', None, bool),
                                  ('x', None, list),
                                  ('y', None, list),
                                  # image_width, image_height
                                  ]:
            if key in kwargs:
                val = kwargs.pop(key)
                val = typ(val)
                args.update({key: val})
            elif default is not None:
                args.update({key: default})
        print('Upload args:', args)
        return args

    def url_upload(self, url, **kwargs):
        args = dict(url=url)
        args.update(self._get_upload_args(**kwargs))
        result = self.send_request('url_upload', args)
        return result

    def upload(self, fn=None, **kwargs):
        args = self._get_upload_args(**kwargs)
        file_args = None
        if fn is not None:
            try:
                f = open(fn, 'rb')
                file_args = (fn, f.read())
            except IOError:
                print('File %s does not exist' % fn)
                raise
        return self.send_request('upload', args, file_args)

    def submission_images(self, subid):
        result = self.send_request('submission_images', {'subid': subid})
        return result.get('image_ids')

    def overlay_plot(self, service, outfn, wcsfn, wcsext=0):
        from astrometry.util import util as anutil
        wcs = anutil.Tan(wcsfn, wcsext)
        params = dict(crval1=wcs.crval[0], crval2=wcs.crval[1],
                      crpix1=wcs.crpix[0], crpix2=wcs.crpix[1],
                      cd11=wcs.cd[0], cd12=wcs.cd[1],
                      cd21=wcs.cd[2], cd22=wcs.cd[3],
                      imagew=wcs.imagew, imageh=wcs.imageh)
        result = self.send_request(service, {'wcs': params})
        print('Result status:', result['status'])
        plotdata = result['plot']
        plotdata = base64.b64decode(plotdata)
        open(outfn, 'wb').write(plotdata)
        print('Wrote', outfn)

    def sdss_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('sdss_image_for_wcs', outfn,
                                 wcsfn, wcsext)

    def galex_plot(self, outfn, wcsfn, wcsext=0):
        return self.overlay_plot('galex_image_for_wcs', outfn,
                                 wcsfn, wcsext)

    def myjobs(self):
        result = self.send_request('myjobs/')
        return result['jobs']

    def job_status(self, job_id, justdict=False):
        result = self.send_request('jobs/%s' % job_id)
        if justdict:
            return result
        stat = result.get('status')
        if stat == 'success':
            result = self.send_request('jobs/%s/calibration' % job_id)
            print('Calibration:', result)
            result = self.send_request('jobs/%s/tags' % job_id)
            print('Tags:', result)
            result = self.send_request('jobs/%s/machine_tags' % job_id)
            print('Machine Tags:', result)
            result = self.send_request('jobs/%s/objects_in_field' % job_id)
            print('Objects in field:', result)
            result = self.send_request('jobs/%s/annotations' % job_id)
            print('Annotations:', result)
            result = self.send_request('jobs/%s/info' % job_id)
            print('Calibration:', result)

        return stat

    def annotate_data(self, job_id):
        """
        :param job_id: id of job
        :return: return data for annotations
        """
        result = self.send_request('jobs/%s/annotations' % job_id)
        return result

    def sub_status(self, sub_id, justdict=False):
        result = self.send_request('submissions/%s' % sub_id)
        if justdict:
            return result
        return result.get('status')

    def jobs_by_tag(self, tag, exact):
        exact_option = 'exact=yes' if exact else ''
        result = self.send_request(
            'jobs_by_tag?query=%s&%s' % (quote(tag.strip()), exact_option),
            {},
        )
        return result


def query_nova(server=Client.default_url, apikey=None, upload=None,
               upload_xy=None, wait=True, wcs=None, newfits=None,
               kmz=None, annotate=None, upload_url=None,
               scale_units=None, scale_lower=None, scale_upper=None,
               scale_est=None, scale_err=None, center_ra=None,
               center_dec=None, radius=None, downsample_factor=None,
               parity=None, tweak_order=None, crpix_center=None,
               sdss_wcs=None, galex_wcs=None, solved_id=None,
               sub_id=None, job_id=None, myjobs=True,
               jobs_by_exact_tag=None, jobs_by_tag=None, public=None,
               allow_mod=None, allow_commercial=None, sleep_interval=5):
    '''
    Parameters
    ----------
    server: str, optional
        Set server base URL (e.g., ``Client.default_url``).

    apikey: str, optional
        API key for Astrometry.net web service;
        if not given will check AN_API_KEY environment variable

    upload: path-like, optional
        Upload a file (The destination path)

    upload_xy: path-like, optional
        Upload a FITS x,y table as JSON

    wait: bool, optional
        After submitting, monitor job status

    wcs: path-like, optional
        Download resulting wcs.fits file, saving to given filename.
        Implies ``wait=True`` if ``urlupload`` or ``upload`` is not None.

    newfits: path-like, optional
        Download resulting new-image.fits file, saving to given filename.
        Implies ``wait=True`` if ``urlupload`` or ``upload`` is not None.

    kmz: path-like, optional
        Download resulting kmz file, saving to given filename;
        Implies ``wait=True`` if ``urlupload`` or ``upload`` is not None.

    annotate: path-like, optional
        Store information about annotations in give file, JSON format;
        Implies ``wait=True`` if ``urlupload`` or ``upload`` is not None.

    upload_url: str, optional
        Upload a file at specified url.

    scale_units: str in ['arcsecperpix', 'arcminwidth', 'degwidth', 'focalmm']
        Units for scale estimate.

    scale_lower, scale_upper: float, optional
        The lower and upper bounds for the size of the image scale. The unit
        is specified by ``scale_units``.

    scale_est: float, optional
        Estimate of the size of the image scale. The unit is specified by
        ``scale_units``.

    scale_err: float, optional
        Scale estimate error (in PERCENT), e.g., 10 if you estimate can be off
        by 10%.

    center_ra, center_dec: float, optional
        RA center and DEC center in the units of degree.

    radius: float, optional
        Search radius around RA, Dec center in the units of degree.

    downsample_factor: int, optional
        Downsample image by this factor.

    parity: str or int in [0, 1, '0', '1'], optional
        Parity (flip) of image.

    tweak_order: int, optional
        SIP distortion order (if None, defaults to 2).

    crpix_center: bool, optional
        Set reference point to center of image?

    sdss_wcs: list of two str, optional
        Plot SDSS image for the given WCS file; write plot to given PNG filename

    galex_wcs: list of two str, optional
        Plot GALEX image for the given WCS file; write plot to given PNG filename

    solved_id: int, optional
        retrieve result for jobId instead of submitting new image

    sub_id, job_id: str, optional
        Get status of a submission or job

    myjobs: bool, optional
        Get all my jobs

    jobs_by_exact_tab: bool, optional
        Get a list of jobs associated with a given tag--exact match

    jobs_by_tag: bool, optional
        Get a list of jobs associated with a given tag

    public: str, optional
        Hide this submission from other users. If `None` (default), code
        ``'y'`` is used, i.e., it is a public submission. Otherwise, code
        ``'n'`` is used, i.e., private submission, and the input value (which
        is not necessarily ``'n'``) will be added to a list called ``args``.

    allow_mod: str, optional
        Select license to allow derivative works of submission, but only if
        shared under same conditions of original license

    allow_commercial: str in ['n', 'd'], optional
        Select license to disallow commercial use of submission

    sleep_interval: int or float, optional
        How long to wait for printing the information.
    '''

    def _change(param, typefunc, initial=None):
        if param is not initial:
            return typefunc(param)
        return param

    def _parse_const(param, default, const, argslist):
        if param is None:
            return default, argslist
        else:
            argslist.append(param)
            return const, argslist

    if apikey is None:
        # try the environment
        apikey = os.environ.get('AN_API_KEY', None)

    if apikey is None:
        raise ValueError(("API key for Astrometry.net web service;"
                          + "if not given will check AN_API_KEY environment variable"
                          + "You must either specify apikey or set AN_API_KEY"))

    accept_units = [None, 'arcsecperpix', 'arcminwidth', 'degwidth', 'focalmm']

    if sdss_wcs is not None:
        if not isinstance(sdss_wcs, list) or len(sdss_wcs) != 2:
            raise ValueError("sdss_wcs must be a list of length 2.")

    if galex_wcs is not None:
        if not isinstance(galex_wcs, list) or len(galex_wcs) != 2:
            raise ValueError("galex_wcs must be a list of length 2.")

    if scale_units not in accept_units:
        raise ValueError(f"scale_units must be one of {accept_units}")

    if parity not in [None, 0, 1, '0', '1']:
        raise ValueError("parity must be one of [None, 0, 1, '0', '1']. ")

    args = []
    public, args = _parse_const(public, 'y', 'n', args)
    allow_mod, args = _parse_const(allow_mod, 'd', 'sa', args)
    allow_commercial, args = _parse_const(allow_commercial, 'd', 'n', args)

    opt = dict(server=_change(server, str),
               upload=_change(upload, str),
               upload_xy=_change(upload_xy, str),
               wait=_change(wait, bool),
               wcs=_change(wcs, str),
               newfits=_change(newfits, str),
               kmz=_change(kmz, str),
               annotate=_change(annotate, str),
               upload_url=_change(upload_url, str),
               scale_units=_change(scale_units, str),
               scale_lower=_change(scale_lower, float),
               scale_upper=_change(scale_upper, float),
               scale_est=_change(scale_est, float),
               scale_err=_change(scale_err, float),
               center_ra=_change(center_ra, float),
               center_dec=_change(center_dec, float),
               radius=_change(radius, float),
               downsample_factor=_change(downsample_factor, int),
               parity=_change(parity, str),
               tweak_order=_change(tweak_order, int),
               crpix_center=_change(crpix_center, bool),
               sdss_wcs=_change(sdss_wcs, list),
               galex_wcs=_change(galex_wcs, list),
               solved_id=_change(solved_id, int),
               sub_id=_change(sub_id, str),
               job_id=_change(job_id, str),
               myjobs=_change(myjobs, bool),
               jobs_by_exact_tag=_change(jobs_by_exact_tag, str),
               jobs_by_tag=_change(jobs_by_tag, str),
               public=public,
               allow_mod=allow_mod,
               allow_commercial=allow_commercial)

    c = Client(apiurl=server)
    c.login(apikey)

    if opt["upload"] or opt["upload_url"] or opt["upload_xy"]:
        if opt["wcs"] or opt["kmz"] or opt["newfits"] or opt["annotate"]:
            opt["wait"] = True

        kwargs = dict(allow_commercial_use=opt["allow_commercial"],
                      allow_modifications=opt["allow_mod"],
                      publicly_visible=opt["public"])
        if opt["scale_lower"] and opt["scale_upper"]:
            kwargs.update(scale_lower=opt["scale_lower"],
                          scale_upper=opt["scale_upper"],
                          scale_type='ul')
        elif opt["scale_est"] and opt["scale_err"]:
            kwargs.update(scale_est=opt["scale_est"],
                          scale_err=opt["scale_err"],
                          scale_type='ev')
        elif opt["scale_lower"] or opt["scale_upper"]:
            kwargs.update(scale_type='ul')
            if scale_lower:
                kwargs.update(scale_lower=scale_lower)
            if scale_upper:
                kwargs.update(scale_upper=scale_upper)

        for key in ['scale_units', 'center_ra', 'center_dec', 'radius',
                    'downsample_factor', 'tweak_order', 'crpix_center', ]:
            if opt[key] is not None:
                kwargs[key] = opt[key]
        if opt["parity"] is not None:
            kwargs.update(parity=int(opt["parity"]))

        if opt["upload"]:
            upres = c.upload(opt["upload"], **kwargs)
        if opt["upload_xy"]:
            from astrometry.util.fits import fits_table
            T = fits_table(opt["upload_xy"])
            kwargs.update(x=[float(x) for x in T.x], y=[float(y) for y in T.y])
            upres = c.upload(**kwargs)
        if opt["upload_url"]:
            upres = c.url_upload(opt["upload_url"], **kwargs)

        stat = upres['status']
        if stat != 'success':
            raise ValueError(f"Upload failed: status, {stat}\n{upres}")

        opt["sub_id"] = upres['subid']

    if opt["wait"]:
        if opt["solved_id"] is None:
            if opt["sub_id"] is None:
                raise ValueError("Can't --wait without a submission id or job id!")

            while True:
                stat = c.sub_status(opt["sub_id"], justdict=True)
                print('Got status:', stat)
                jobs = stat.get('jobs', [])
                if len(jobs):
                    for j in jobs:
                        if j is not None:
                            break
                    if j is not None:
                        print('Selecting job id', j)
                        opt["solved_id"] = j
                        break
                time.sleep(sleep_interval)

        while True:
            stat = c.job_status(opt["solved_id"], justdict=True)
            print('Got job status:', stat)
            if stat.get('status', '') in ['success']:
                success = (stat['status'] == 'success')
                break
            time.sleep(sleep_interval)

    if opt["solved_id"]:
        # we have a jobId for retrieving results
        retrieveurls = []
        if opt["wcs"]:
            # We don't need the API for this, just construct URL
            url = opt["server"].replace(
                '/api/', '/wcs_file/%i' % opt["solved_id"])
            retrieveurls.append((url, opt["wcs"]))
        if opt["kmz"]:
            url = opt["server"].replace(
                '/api/', '/kml_file/%i/' % opt["solved_id"])
            retrieveurls.append((url, opt["kmz"]))
        if opt["newfits"]:
            url = opt["server"].replace(
                '/api/', '/new_fits_file/%i/' % opt["solved_id"])
            retrieveurls.append((url, opt["newfits"]))

        for url, fn in retrieveurls:
            print('Retrieving file from', url, 'to', fn)
            f = urlopen(url)
            txt = f.read()
            w = open(fn, 'wb')
            w.write(txt)
            w.close()
            print('Wrote to', fn)

        if opt["annotate"]:
            result = c.annotate_data(opt["solved_id"])
            with open(opt["annotate"], 'w') as f:
                f.write(python2json(result))

    if opt["wait"]:
        # behaviour as in old implementation
        opt["sub_id"] = None

    if opt["sdss_wcs"]:
        (wcsfn, outfn) = opt["sdss_wcs"]
        c.sdss_plot(outfn, wcsfn)
    if opt["galex_wcs"]:
        (wcsfn, outfn) = opt["galex_wcs"]
        c.galex_plot(outfn, wcsfn)

    if opt["sub_id"]:
        print(c.sub_status(opt["sub_id"]))
    if opt["job_id"]:
        print(c.job_status(opt["job_id"]))

    if opt["jobs_by_tag"]:
        tag = opt["jobs_by_tag"]
        print(c.jobs_by_tag(tag, None))
    if opt["jobs_by_exact_tag"]:
        tag = opt["jobs_by_exact_tag"]
        print(c.jobs_by_tag(tag, 'yes'))

    if opt["myjobs"]:
        print(jobs)
        jobs = c.myjobs()

    return success
