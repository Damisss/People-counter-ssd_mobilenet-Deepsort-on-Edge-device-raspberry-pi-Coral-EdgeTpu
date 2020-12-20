class TrackableOBject ():
  def __init__(self, objectId, centroid):
    try:
      self.objectId = objectId
      self.centroids = [centroid]
      self.counted = False
    
    except Exception as e:
      raise e